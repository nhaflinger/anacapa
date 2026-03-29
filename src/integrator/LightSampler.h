#pragma once

#include <anacapa/shading/ILight.h>
#include <anacapa/sampling/ISampler.h>
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// LightSampler — alias table for O(1) light selection weighted by power
//
// Build once from the scene's light list. At render time, a single
// get1D() sample selects a light in O(1) with probability proportional
// to its estimated power. The selection PDF is returned so callers can
// divide it out of the contribution.
//
// Reference: Vose's alias method (Walker 1977)
// ---------------------------------------------------------------------------
class LightSampler {
public:
    struct Selection {
        const ILight* light = nullptr;
        uint32_t      index = 0;
        float         pdf   = 0.f;   // Probability of selecting this light
    };

    void build(const std::vector<const ILight*>& lights) {
        m_lights = lights;
        uint32_t n = static_cast<uint32_t>(lights.size());
        if (n == 0) return;

        // Compute normalized weights from power estimates
        std::vector<float> weights(n);
        float total = 0.f;
        for (uint32_t i = 0; i < n; ++i) {
            weights[i] = std::max(0.f, lights[i]->power());
            total += weights[i];
        }
        // Fall back to uniform if all powers are zero
        if (total <= 0.f) {
            for (auto& w : weights) w = 1.f;
            total = static_cast<float>(n);
        }
        float invTotal = 1.f / total;
        for (auto& w : weights) w *= invTotal;

        // Build alias table
        m_prob.resize(n);
        m_alias.resize(n);

        std::vector<uint32_t> small, large;
        for (uint32_t i = 0; i < n; ++i) {
            m_prob[i] = weights[i] * n;
            (m_prob[i] < 1.f ? small : large).push_back(i);
        }
        while (!small.empty() && !large.empty()) {
            uint32_t s = small.back(); small.pop_back();
            uint32_t l = large.back(); large.pop_back();
            m_alias[s] = l;
            m_prob[l]  = m_prob[l] + m_prob[s] - 1.f;
            (m_prob[l] < 1.f ? small : large).push_back(l);
        }
        // Floating-point residuals
        for (uint32_t i : small) m_prob[i] = 1.f;
        for (uint32_t i : large) m_prob[i] = 1.f;

        // Store the original normalized weights for pdf() queries
        m_weights = weights;
    }

    // Select a light using a uniform sample u in [0,1)
    Selection sample(float u) const {
        if (m_lights.empty()) return {};
        uint32_t n  = static_cast<uint32_t>(m_lights.size());
        uint32_t i  = std::min(static_cast<uint32_t>(u * n), n - 1);
        float    uf = u * n - static_cast<float>(i);
        uint32_t idx = (uf < m_prob[i]) ? i : m_alias[i];
        return { m_lights[idx], idx, m_weights[idx] };
    }

    float pdf(uint32_t lightIndex) const {
        if (lightIndex >= m_weights.size()) return 0.f;
        return m_weights[lightIndex];
    }

    uint32_t size() const { return static_cast<uint32_t>(m_lights.size()); }
    bool     empty() const { return m_lights.empty(); }

private:
    std::vector<const ILight*> m_lights;
    std::vector<float>         m_prob;     // Alias table probability column
    std::vector<uint32_t>      m_alias;    // Alias table alias column
    std::vector<float>         m_weights;  // Normalized selection PDFs
};

} // namespace anacapa
