#include "mx_funcs.h"

#define true 1
#define false 0
struct textureresource { string filename; string colorspace; };
#define BSDF closure color
#define EDF closure color
#define VDF closure color
struct surfaceshader { closure color bsdf; closure color edf; float opacity; };
#define volumeshader closure color
#define displacementshader vector
#define lightshader closure color
#define MATERIAL closure color

#define M_FLOAT_EPS 1e-8
closure color null_closure() { closure color null_closure = 0; return null_closure; } 

void NG_open_pbr_anisotropy(float roughness, float anisotropy, output vector2 out)
{
    float rough_sq_out = roughness * roughness;
    float aniso_invert_amount_tmp = 1.000000;
    float aniso_invert_out = aniso_invert_amount_tmp - anisotropy;
    float aniso_invert_sq_out = aniso_invert_out * aniso_invert_out;
    float denom_in2_tmp = 1.000000;
    float denom_out = aniso_invert_sq_out + denom_in2_tmp;
    float fraction_in1_tmp = 2.000000;
    float fraction_out = fraction_in1_tmp / denom_out;
    float sqrt_out = sqrt(fraction_out);
    float alpha_x_out = rough_sq_out * sqrt_out;
    float alpha_y_out = aniso_invert_out * alpha_x_out;
    vector2 result_out = { alpha_x_out,alpha_y_out };
    out = result_out;
}

void NG_separate3_color3(color in, output float outr, output float outg, output float outb)
{
    int N_extract_0_index_tmp = 0;
    float N_extract_0_out = mx_extract(in, N_extract_0_index_tmp);
    int N_extract_1_index_tmp = 1;
    float N_extract_1_out = mx_extract(in, N_extract_1_index_tmp);
    int N_extract_2_index_tmp = 2;
    float N_extract_2_out = mx_extract(in, N_extract_2_index_tmp);
    outr = N_extract_0_out;
    outg = N_extract_1_out;
    outb = N_extract_2_out;
}

void NG_convert_color3_vector3(color in, output vector out)
{
    float separate_outr = 0.0;
    float separate_outg = 0.0;
    float separate_outb = 0.0;
    NG_separate3_color3(in, separate_outr, separate_outg, separate_outb);
    vector combine_out = vector( separate_outr,separate_outg,separate_outb );
    out = combine_out;
}

void NG_convert_float_vector3(float in, output vector out)
{
    vector combine_out = vector( in,in,in );
    out = combine_out;
}

void mx_dielectric_bsdf(float weight, color tint, float ior, vector2 roughness, float thinfilm_thickness, float thinfilm_ior, normal N, vector U, string distribution, string scatter_mode, output BSDF bsdf)
{
    color reflection_tint = (scatter_mode == "T") ? color(0.0) : tint;
    color transmission_tint = (scatter_mode == "R") ? color(0.0) : tint;
    bsdf = weight * dielectric_bsdf(N, U, reflection_tint, transmission_tint, roughness.x, roughness.y, ior, distribution, "thinfilm_thickness", thinfilm_thickness, "thinfilm_ior", thinfilm_ior);
}

void mx_subsurface_bsdf(float weight, color albedo, color radius, float anisotropy, normal N, output BSDF bsdf)
{
#if OSL_VERSION_MAJOR >= 1 && OSL_VERSION_MINOR >= 14
    bsdf = weight * subsurface_bssrdf(N, albedo, radius, anisotropy);
#else
    bsdf = weight * subsurface_bssrdf(N, albedo, 1.0, radius, anisotropy);
#endif
}

void NG_convert_float_color3(float in, output color out)
{
    color combine_out = color( in,in,in );
    out = combine_out;
}

float mx_square(float x)
{
    return x*x;
}

vector2 mx_square(vector2 x)
{
    return x*x;
}

vector mx_square(vector x)
{
    return x*x;
}

vector4 mx_square(vector4 x)
{
    return x*x;
}

float mx_pow5(float x)
{
    return mx_square(mx_square(x)) * x;
}

color mx_fresnel_conductor(float cosTheta, vector n, vector k)
{
   float c2 = cosTheta*cosTheta;
   vector n2_k2 = n*n + k*k;
   vector nc2 = 2.0 * n * cosTheta;

   vector rs_a = n2_k2 + c2;
   vector rp_a = n2_k2 * c2 + 1.0;
   vector rs = (rs_a - nc2) / (rs_a + nc2);
   vector rp = (rp_a - nc2) / (rp_a + nc2);

   return 0.5 * (rs + rp);
}

// Standard Schlick Fresnel
float mx_fresnel_schlick(float cosTheta, float F0)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return F0 + (1.0 - F0) * x5;
}
color mx_fresnel_schlick(float cosTheta, color F0)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return F0 + (1.0 - F0) * x5;
}

// Generalized Schlick Fresnel
float mx_fresnel_schlick(float cosTheta, float F0, float F90)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return mix(F0, F90, x5);
}
color mx_fresnel_schlick(float cosTheta, color F0, color F90)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return mix(F0, F90, x5);
}

// Generalized Schlick Fresnel with a variable exponent
color mx_fresnel_schlick(float cosTheta, float f0, float f90, float exponent)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    return mix(f0, f90, pow(x, exponent));
}
color mx_fresnel_schlick(float cosTheta, color f0, color f90, float exponent)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    return mix(f0, f90, pow(x, exponent));
}

void mx_generalized_schlick_edf(color color0, color color90, float exponent, EDF base, output EDF result)
{
    float NdotV = fabs(dot(N,-I));
    color f = mx_fresnel_schlick(NdotV, color0, color90, exponent);
    result = base * f;
}

void mx_generalized_schlick_bsdf(float weight, color color0, color color82, color color90, float exponent, vector2 roughness, float thinfilm_thickness, float thinfilm_ior, normal N, vector U, string distribution, string scatter_mode, output BSDF bsdf)
{
    color reflection_tint = (scatter_mode == "T") ? color(0.0) : color(1.0);
    color transmission_tint = (scatter_mode == "R") ? color(0.0) : color(1.0);
    bsdf = weight * generalized_schlick_bsdf(N, U, reflection_tint, transmission_tint, roughness.x, roughness.y, color0, color90, exponent, distribution, "thinfilm_thickness", thinfilm_thickness, "thinfilm_ior", thinfilm_ior);
}

void mx_anisotropic_vdf(color absorption, color scattering, float anisotropy, output VDF vdf)
{
    // Convert from absorption and scattering coefficients to
    // extinction coefficient and single-scattering albedo.
    color extinction = absorption + scattering;
    color albedo = scattering / extinction;
    vdf = anisotropic_vdf(albedo, extinction, anisotropy);
}

void mx_surface(BSDF bsdf, EDF edf, float opacity, int thin_walled, output surfaceshader result)
{
    result.bsdf    = bsdf;
    result.edf     = edf;
    result.opacity = clamp(opacity, 0.0, 1.0);
}

void NG_open_pbr_surface_surfaceshader(float base_weight, color base_color, float base_diffuse_roughness, float base_metalness, float specular_weight, color specular_color, float specular_roughness, float specular_ior, float specular_roughness_anisotropy, float transmission_weight, color transmission_color, float transmission_depth, color transmission_scatter, float transmission_scatter_anisotropy, float transmission_dispersion_scale, float transmission_dispersion_abbe_number, float subsurface_weight, color subsurface_color, float subsurface_radius, color subsurface_radius_scale, float subsurface_scatter_anisotropy, float fuzz_weight, color fuzz_color, float fuzz_roughness, float coat_weight, color coat_color, float coat_roughness, float coat_roughness_anisotropy, float coat_ior, float coat_darkening, float thin_film_weight, float thin_film_thickness, float thin_film_ior, float emission_luminance, color emission_color, float geometry_opacity, int geometry_thin_walled, vector geometry_normal, vector geometry_coat_normal, vector geometry_tangent, vector geometry_coat_tangent, output surfaceshader out)
{
    vector2 coat_roughness_vector_out = vector2(0.0, 0.0);
    NG_open_pbr_anisotropy(coat_roughness, coat_roughness_anisotropy, coat_roughness_vector_out);
    color metal_reflectivity_out = base_color * base_weight;
    color metal_edgecolor_out = specular_color * specular_weight;
    float coat_roughness_to_power_4_in2_tmp = 4.000000;
    float coat_roughness_to_power_4_out = pow(coat_roughness, coat_roughness_to_power_4_in2_tmp);
    float specular_roughness_to_power_4_in2_tmp = 4.000000;
    float specular_roughness_to_power_4_out = pow(specular_roughness, specular_roughness_to_power_4_in2_tmp);
    float thin_film_thickness_nm_in2_tmp = 1000.000000;
    float thin_film_thickness_nm_out = thin_film_thickness * thin_film_thickness_nm_in2_tmp;
    float specular_to_coat_ior_ratio_out = specular_ior / coat_ior;
    float coat_to_specular_ior_ratio_out = coat_ior / specular_ior;
    float if_transmission_tint_value2_tmp = 0.000000;
    color if_transmission_tint_in1_tmp = color(1.000000, 1.000000, 1.000000);
    color if_transmission_tint_out = mx_ternary(transmission_depth > if_transmission_tint_value2_tmp, if_transmission_tint_in1_tmp, transmission_color);
    vector transmission_color_vector_out = vector(0.0);
    NG_convert_color3_vector3(transmission_color, transmission_color_vector_out);
    vector transmission_depth_vector_out = vector(0.0);
    NG_convert_float_vector3(transmission_depth, transmission_depth_vector_out);
    vector transmission_scatter_vector_out = vector(0.0);
    NG_convert_color3_vector3(transmission_scatter, transmission_scatter_vector_out);
    float subsurface_color_nonnegative_in2_tmp = 0.000000;
    color subsurface_color_nonnegative_out = max(subsurface_color, subsurface_color_nonnegative_in2_tmp);
    float one_minus_subsurface_scatter_anisotropy_in1_tmp = 1.000000;
    float one_minus_subsurface_scatter_anisotropy_out = one_minus_subsurface_scatter_anisotropy_in1_tmp - subsurface_scatter_anisotropy;
    float one_plus_subsurface_scatter_anisotropy_in1_tmp = 1.000000;
    float one_plus_subsurface_scatter_anisotropy_out = one_plus_subsurface_scatter_anisotropy_in1_tmp + subsurface_scatter_anisotropy;
    color subsurface_radius_scaled_out = subsurface_radius_scale * subsurface_radius;
    float subsurface_selector_out = float(geometry_thin_walled);
    float base_color_nonnegative_in2_tmp = 0.000000;
    color base_color_nonnegative_out = max(base_color, base_color_nonnegative_in2_tmp);
    float coat_ior_minus_one_in2_tmp = 1.000000;
    float coat_ior_minus_one_out = coat_ior - coat_ior_minus_one_in2_tmp;
    float coat_ior_plus_one_in1_tmp = 1.000000;
    float coat_ior_plus_one_out = coat_ior_plus_one_in1_tmp + coat_ior;
    float coat_ior_sqr_out = coat_ior * coat_ior;
    color Emetal_out = base_color * specular_weight;
    color Edielectric_out = mix(base_color, subsurface_color, subsurface_weight);
    float coat_weight_times_coat_darkening_out = coat_weight * coat_darkening;
    color coat_attenuation_bg_tmp = color(1.000000, 1.000000, 1.000000);
    color coat_attenuation_out = mix(coat_attenuation_bg_tmp, coat_color, coat_weight);
    color emission_weight_out = emission_color * emission_luminance;
    float two_times_coat_roughness_to_power_4_in2_tmp = 2.000000;
    float two_times_coat_roughness_to_power_4_out = coat_roughness_to_power_4_out * two_times_coat_roughness_to_power_4_in2_tmp;
    float specular_to_coat_ior_ratio_tir_fix_value2_tmp = 1.000000;
    float specular_to_coat_ior_ratio_tir_fix_out = mx_ternary(specular_to_coat_ior_ratio_out > specular_to_coat_ior_ratio_tir_fix_value2_tmp, specular_to_coat_ior_ratio_out, coat_to_specular_ior_ratio_out);
    vector transmission_color_ln_out = log(transmission_color_vector_out);
    vector scattering_coeff_out = transmission_scatter_vector_out / transmission_depth_vector_out;
    color subsurface_thin_walled_brdf_factor_out = subsurface_color * one_minus_subsurface_scatter_anisotropy_out;
    color subsurface_thin_walled_btdf_factor_out = subsurface_color * one_plus_subsurface_scatter_anisotropy_out;
    float coat_ior_to_F0_sqrt_out = coat_ior_minus_one_out / coat_ior_plus_one_out;
    color Ebase_out = mix(Edielectric_out, Emetal_out, base_metalness);
    float add_coat_and_spec_roughnesses_to_power_4_out = two_times_coat_roughness_to_power_4_out + specular_roughness_to_power_4_out;
    float eta_s_out = mix(specular_ior, specular_to_coat_ior_ratio_tir_fix_out, coat_weight);
    float extinction_coeff_denom_in2_tmp = -1.000000;
    vector extinction_coeff_denom_out = transmission_color_ln_out * extinction_coeff_denom_in2_tmp;
    float if_volume_scattering_value2_tmp = 0.000000;
    vector if_volume_scattering_in2_tmp = vector(0.000000, 0.000000, 0.000000);
    vector if_volume_scattering_out = mx_ternary(transmission_depth > if_volume_scattering_value2_tmp, scattering_coeff_out, if_volume_scattering_in2_tmp);
    float coat_ior_to_F0_out = coat_ior_to_F0_sqrt_out * coat_ior_to_F0_sqrt_out;
    float min_1_add_coat_and_spec_roughnesses_to_power_4_in1_tmp = 1.000000;
    float min_1_add_coat_and_spec_roughnesses_to_power_4_out = min(min_1_add_coat_and_spec_roughnesses_to_power_4_in1_tmp, add_coat_and_spec_roughnesses_to_power_4_out);
    float eta_s_minus_one_in2_tmp = 1.000000;
    float eta_s_minus_one_out = eta_s_out - eta_s_minus_one_in2_tmp;
    float eta_s_plus_one_in2_tmp = 1.000000;
    float eta_s_plus_one_out = eta_s_out + eta_s_plus_one_in2_tmp;
    vector extinction_coeff_out = extinction_coeff_denom_out / transmission_depth_vector_out;
    float one_minus_coat_F0_in1_tmp = 1.000000;
    float one_minus_coat_F0_out = one_minus_coat_F0_in1_tmp - coat_ior_to_F0_out;
    float coat_affected_specular_roughness_in2_tmp = 0.250000;
    float coat_affected_specular_roughness_out = pow(min_1_add_coat_and_spec_roughnesses_to_power_4_out, coat_affected_specular_roughness_in2_tmp);
    float sign_eta_s_minus_one_out = sign(eta_s_minus_one_out);
    float specular_F0_sqrt_out = eta_s_minus_one_out / eta_s_plus_one_out;
    vector absorption_coeff_out = extinction_coeff_out - scattering_coeff_out;
    float one_minus_coat_F0_over_eta2_out = one_minus_coat_F0_out / coat_ior_sqr_out;
    color one_minus_coat_F0_color_out = color(0.0);
    NG_convert_float_color3(one_minus_coat_F0_out, one_minus_coat_F0_color_out);
    float effective_specular_roughness_out = mix(specular_roughness, coat_affected_specular_roughness_out, coat_weight);
    float specular_F0_out = specular_F0_sqrt_out * specular_F0_sqrt_out;
    int absorption_coeff_x_index_tmp = 0;
    float absorption_coeff_x_out = mx_extract(absorption_coeff_out, absorption_coeff_x_index_tmp);
    int absorption_coeff_y_index_tmp = 1;
    float absorption_coeff_y_out = mx_extract(absorption_coeff_out, absorption_coeff_y_index_tmp);
    int absorption_coeff_z_index_tmp = 2;
    float absorption_coeff_z_out = mx_extract(absorption_coeff_out, absorption_coeff_z_index_tmp);
    float Kcoat_in1_tmp = 1.000000;
    float Kcoat_out = Kcoat_in1_tmp - one_minus_coat_F0_over_eta2_out;
    vector2 main_roughness_out = vector2(0.0, 0.0);
    NG_open_pbr_anisotropy(effective_specular_roughness_out, specular_roughness_anisotropy, main_roughness_out);
    float scaled_specular_F0_out = specular_weight * specular_F0_out;
    float absorption_coeff_min_x_y_out = min(absorption_coeff_x_out, absorption_coeff_y_out);
    float one_minus_Kcoat_in1_tmp = 1.000000;
    float one_minus_Kcoat_out = one_minus_Kcoat_in1_tmp - Kcoat_out;
    color Ebase_Kcoat_out = Ebase_out * Kcoat_out;
    float scaled_specular_F0_clamped_low_tmp = 0.000000;
    float scaled_specular_F0_clamped_high_tmp = 0.999990;
    float scaled_specular_F0_clamped_out = clamp(scaled_specular_F0_out, scaled_specular_F0_clamped_low_tmp, scaled_specular_F0_clamped_high_tmp);
    float absorption_coeff_min_out = min(absorption_coeff_min_x_y_out, absorption_coeff_z_out);
    color one_minus_Kcoat_color_out = color(0.0);
    NG_convert_float_color3(one_minus_Kcoat_out, one_minus_Kcoat_color_out);
    color one_minus_Ebase_Kcoat_in1_tmp = color(1.000000, 1.000000, 1.000000);
    color one_minus_Ebase_Kcoat_out = one_minus_Ebase_Kcoat_in1_tmp - Ebase_Kcoat_out;
    float sqrt_scaled_specular_F0_out = sqrt(scaled_specular_F0_clamped_out);
    vector absorption_coeff_min_vector_out = vector(0.0);
    NG_convert_float_vector3(absorption_coeff_min_out, absorption_coeff_min_vector_out);
    color base_darkening_out = one_minus_Kcoat_color_out / one_minus_Ebase_Kcoat_out;
    float modulated_eta_s_epsilon_out = sign_eta_s_minus_one_out * sqrt_scaled_specular_F0_out;
    vector absorption_coeff_shifted_out = absorption_coeff_out - absorption_coeff_min_vector_out;
    color modulated_base_darkening_bg_tmp = color(1.000000, 1.000000, 1.000000);
    color modulated_base_darkening_out = mix(modulated_base_darkening_bg_tmp, base_darkening_out, coat_weight_times_coat_darkening_out);
    float one_plus_modulated_eta_s_epsilon_in1_tmp = 1.000000;
    float one_plus_modulated_eta_s_epsilon_out = one_plus_modulated_eta_s_epsilon_in1_tmp + modulated_eta_s_epsilon_out;
    float one_minus_modulated_eta_s_epsilon_in1_tmp = 1.000000;
    float one_minus_modulated_eta_s_epsilon_out = one_minus_modulated_eta_s_epsilon_in1_tmp - modulated_eta_s_epsilon_out;
    float if_absorption_coeff_shifted_value1_tmp = 0.000000;
    vector if_absorption_coeff_shifted_out = mx_ternary(if_absorption_coeff_shifted_value1_tmp > absorption_coeff_min_out, absorption_coeff_shifted_out, absorption_coeff_out);
    float modulated_eta_s_out = one_plus_modulated_eta_s_epsilon_out / one_minus_modulated_eta_s_epsilon_out;
    float if_volume_absorption_value2_tmp = 0.000000;
    vector if_volume_absorption_in2_tmp = vector(0.000000, 0.000000, 0.000000);
    vector if_volume_absorption_out = mx_ternary(transmission_depth > if_volume_absorption_value2_tmp, if_absorption_coeff_shifted_out, if_volume_absorption_in2_tmp);
    BSDF fuzz_bsdf_out = fuzz_weight * sheen_bsdf(geometry_normal, fuzz_color, fuzz_roughness);
    BSDF coat_bsdf_out = null_closure();
    mx_dielectric_bsdf(coat_weight, color(1.000000, 1.000000, 1.000000), coat_ior, coat_roughness_vector_out, 0.000000, 1.500000, geometry_coat_normal, geometry_coat_tangent, "ggx", "R", coat_bsdf_out);
    BSDF metal_bsdf_tf_out = null_closure();
    mx_generalized_schlick_bsdf(specular_weight, metal_reflectivity_out, metal_edgecolor_out, color(1.000000, 1.000000, 1.000000), 5.000000, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, "ggx", "R", metal_bsdf_tf_out);
    BSDF metal_bsdf_out = null_closure();
    mx_generalized_schlick_bsdf(specular_weight, metal_reflectivity_out, metal_edgecolor_out, color(1.000000, 1.000000, 1.000000), 5.000000, main_roughness_out, 0.000000, 1.500000, geometry_normal, geometry_tangent, "ggx", "R", metal_bsdf_out);
    BSDF metal_bsdf_tf_mix_out = mix(metal_bsdf_out, metal_bsdf_tf_out, thin_film_weight);
    BSDF dielectric_reflection_tf_out = null_closure();
    mx_dielectric_bsdf(1.000000, specular_color, modulated_eta_s_out, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, "ggx", "R", dielectric_reflection_tf_out);
    BSDF dielectric_reflection_out = null_closure();
    mx_dielectric_bsdf(1.000000, specular_color, modulated_eta_s_out, main_roughness_out, 0.000000, 1.500000, geometry_normal, geometry_tangent, "ggx", "R", dielectric_reflection_out);
    BSDF dielectric_reflection_tf_mix_out = mix(dielectric_reflection_out, dielectric_reflection_tf_out, thin_film_weight);
    BSDF dielectric_transmission_out = null_closure();
    mx_dielectric_bsdf(1.000000, if_transmission_tint_out, modulated_eta_s_out, main_roughness_out, 0.000000, 1.500000, geometry_normal, geometry_tangent, "ggx", "T", dielectric_transmission_out);
    VDF dielectric_volume_out = null_closure();
    mx_anisotropic_vdf(if_volume_absorption_out, if_volume_scattering_out, transmission_scatter_anisotropy, dielectric_volume_out);
    BSDF dielectric_volume_transmission_out = layer(dielectric_transmission_out, dielectric_volume_out);
    float subsurface_thin_walled_reflection_bsdf_weight_tmp = 1.000000;
    BSDF subsurface_thin_walled_reflection_bsdf_out = subsurface_thin_walled_reflection_bsdf_weight_tmp * oren_nayar_diffuse_bsdf(geometry_normal, subsurface_color_nonnegative_out, base_diffuse_roughness);
    BSDF subsurface_thin_walled_reflection_out = (subsurface_thin_walled_brdf_factor_out * subsurface_thin_walled_reflection_bsdf_out);
    float subsurface_thin_walled_transmission_bsdf_weight_tmp = 1.000000;
    BSDF subsurface_thin_walled_transmission_bsdf_out = subsurface_thin_walled_transmission_bsdf_weight_tmp * translucent_bsdf(geometry_normal, subsurface_color_nonnegative_out);
    BSDF subsurface_thin_walled_transmission_out = (subsurface_thin_walled_btdf_factor_out * subsurface_thin_walled_transmission_bsdf_out);
    float subsurface_thin_walled_mix_tmp = 0.500000;
    BSDF subsurface_thin_walled_out = mix(subsurface_thin_walled_transmission_out, subsurface_thin_walled_reflection_out, subsurface_thin_walled_mix_tmp);
    BSDF subsurface_bsdf_out = null_closure();
    mx_subsurface_bsdf(1.000000, subsurface_color_nonnegative_out, subsurface_radius_scaled_out, subsurface_scatter_anisotropy, geometry_normal, subsurface_bsdf_out);
    BSDF selected_subsurface_out = mix(subsurface_bsdf_out, subsurface_thin_walled_out, subsurface_selector_out);
    BSDF diffuse_bsdf_out = base_weight * oren_nayar_diffuse_bsdf(geometry_normal, base_color_nonnegative_out, base_diffuse_roughness);
    BSDF opaque_base_out = mix(diffuse_bsdf_out, selected_subsurface_out, subsurface_weight);
    BSDF dielectric_substrate_out = mix(opaque_base_out, dielectric_volume_transmission_out, transmission_weight);
    BSDF dielectric_base_out = layer(dielectric_reflection_tf_mix_out, dielectric_substrate_out);
    BSDF base_substrate_out = mix(dielectric_base_out, metal_bsdf_tf_mix_out, base_metalness);
    BSDF darkened_base_substrate_out = (modulated_base_darkening_out * base_substrate_out);
    BSDF coat_substrate_attenuated_out = (coat_attenuation_out * darkened_base_substrate_out);
    BSDF coat_layer_out = layer(coat_bsdf_out, coat_substrate_attenuated_out);
    BSDF fuzz_layer_out = layer(fuzz_bsdf_out, coat_layer_out);
    EDF uncoated_emission_edf_out = uniform_edf(emission_weight_out);
    EDF coat_tinted_emission_edf_out = (coat_color * uncoated_emission_edf_out);
    EDF coated_emission_edf_out = null_closure();
    mx_generalized_schlick_edf(one_minus_coat_F0_color_out, color(0.000000, 0.000000, 0.000000), 5.000000, coat_tinted_emission_edf_out, coated_emission_edf_out);
    EDF emission_edf_out = mix(uncoated_emission_edf_out, coated_emission_edf_out, coat_weight);
    surfaceshader shader_constructor_out = surfaceshader(null_closure(), null_closure(), 1.0);
    mx_surface(fuzz_layer_out, emission_edf_out, geometry_opacity, 0, shader_constructor_out);
    out = shader_constructor_out;
}

void mx_surfacematerial(surfaceshader surface, surfaceshader back, displacementshader disp, output MATERIAL result)
{
    float opacity_weight = clamp(surface.opacity, 0.0, 1.0);
    result =  (surface.bsdf + surface.edf) * opacity_weight + transparent() * (1.0 - opacity_weight);
}


// ---- mx_*.osl function definitions ----
// --- lib/mx_transform_uv.osl ---
vector2 mx_transform_uv(vector2 texcoord)
{
    return texcoord;
}

// --- lib/vector4_extra_ops.osl ---
// Adds some syntactic sugar allowing mixing vector4 and color4 as
// arguments of some binary operators used by OCIO transform code.

vector4 __operator__mul__(matrix m, vector4 v)
{
    return transform(m, v);
}

vector4 __operator__mul__(color4 c, vector4 v)
{
    return vector4(c.rgb.r, c.rgb.g, c.rgb.b, c.a) * v;
}

vector4 __operator__mul__(vector4 v, color4 c)
{
    return c * v;
}

vector4 __operator__sub__(color4 c, vector4 v)
{
    return vector4(c.rgb.r, c.rgb.g, c.rgb.b, c.a) - v;
}

vector4 __operator__add__(vector4 v, color4 c)
{
    return v + vector4(c.rgb.r, c.rgb.g, c.rgb.b, c.a);
}

vector4 __operator__add__(color4 c, vector4 v)
{
    return v + c;
}

vector4 pow(color4 c, vector4 v)
{
    return pow(vector4(c.rgb.r, c.rgb.g, c.rgb.b, c.a), v);
}

vector4 max(vector4 v, color4 c)
{
    return max(v, vector4(c.rgb.r, c.rgb.g, c.rgb.b, c.a));
}

// --- mx_burn_float.osl ---
void mx_burn_float(float fg, float bg, float mix, output float result)
{
    if (abs(fg) < M_FLOAT_EPS)
    {
        result = 0.0;
        return;
    }
    result = mix*(1.0 - ((1.0 - bg) / fg)) + ((1.0-mix)*bg);
}

// --- mx_burn_color3.osl ---

void mx_burn_color3(color fg, color bg, float mix, output color result)
{
    mx_burn_float(fg[0], bg[0], mix, result[0]);
    mx_burn_float(fg[1], bg[1], mix, result[1]);
    mx_burn_float(fg[2], bg[2], mix, result[2]);
}

// --- mx_burn_color4.osl ---

void mx_burn_color4(color4 fg, color4 bg, float mix, output color4 result)
{
    mx_burn_float(fg.rgb[0], bg.rgb[0], mix, result.rgb[0]);
    mx_burn_float(fg.rgb[1], bg.rgb[1], mix, result.rgb[1]);
    mx_burn_float(fg.rgb[2], bg.rgb[2], mix, result.rgb[2]);
    mx_burn_float(fg.a, bg.a, mix, result.a);
}

// --- mx_cellnoise2d_float.osl ---
void mx_cellnoise2d_float(vector2 texcoord, output float result)
{
    result = cellnoise(texcoord.x, texcoord.y);
}

// --- mx_cellnoise3d_float.osl ---
void mx_cellnoise3d_float(vector position, output float result)
{
    result = cellnoise(position);
}

// --- mx_creatematrix.osl ---
void mx_creatematrix_vector3_matrix33(vector in1, vector in2, vector in3, output matrix result)
{
    result = matrix(in1.x, in1.y, in1.z,  0.0,
                    in2.x, in2.y, in2.z,  0.0,
                    in3.x, in3.y, in3.z,  0.0,
                    0.0,   0.0,   0.0,    1.0);
}

void mx_creatematrix_vector3_matrix44(vector in1, vector in2, vector in3, vector in4, output matrix result)
{
    result = matrix(in1.x, in1.y, in1.z,  0.0,
                    in2.x, in2.y, in2.z,  0.0,
                    in3.x, in3.y, in3.z,  0.0,
                    in4.x, in4.y, in4.z,  1.0);
}

void mx_creatematrix_vector4_matrix44(vector4 in1, vector4 in2, vector4 in3, vector4 in4, output matrix result)
{
    result = matrix(in1.x, in1.y, in1.z,  in1.w,
                    in2.x, in2.y, in2.z,  in2.w,
                    in3.x, in3.y, in3.z,  in3.w,
                    in4.x, in4.y, in4.z,  in4.w);
}

// --- mx_disjointover_color4.osl ---
void mx_disjointover_color4(color4 fg, color4 bg, float mix, output color4 result)
{
    float summedAlpha = fg.a + bg.a;

    if (summedAlpha <= 1)
    {
        result.rgb = fg.rgb + bg.rgb;
    }
    else
    {
        if (abs(bg.a) < M_FLOAT_EPS)
        {
            result.rgb = 0.0;
        }
        else
        {
            float x = (1 - fg.a) / bg.a;
            result.rgb = fg.rgb + bg.rgb * x;
        }
    }
    result.a = min(summedAlpha, 1.0);

    result.rgb = result.rgb * mix + (1.0 - mix) * bg.rgb;
    result.a = result.a * mix + (1.0 - mix) * bg.a;
}

// --- mx_dodge_float.osl ---
void mx_dodge_float(float fg, float bg, float mix, output float out)
{
    if (abs(1.0 - fg) < M_FLOAT_EPS)
    {
        out = 0.0;
        return;
    }
    out = mix*(bg / (1.0 - fg)) + ((1.0-mix)*bg);
}

// --- mx_dodge_color3.osl ---

void mx_dodge_color3(color fg, color bg, float mix, output color result)
{
    mx_dodge_float(fg[0], bg[0], mix, result[0]);
    mx_dodge_float(fg[1], bg[1], mix, result[1]);
    mx_dodge_float(fg[2], bg[2], mix, result[2]);
}

// --- mx_dodge_color4.osl ---

void mx_dodge_color4(color4 fg , color4 bg , float mix , output color4 result)
{
    mx_dodge_float(fg.rgb[0], bg.rgb[0], mix, result.rgb[0]);
    mx_dodge_float(fg.rgb[1], bg.rgb[1], mix, result.rgb[1]);
    mx_dodge_float(fg.rgb[2], bg.rgb[2], mix, result.rgb[2]);
    mx_dodge_float(fg.a, bg.a, mix, result.a);
}

// --- mx_fractal2d_float.osl ---
void mx_fractal2d_float(float amplitude, int octaves, float lacunarity, float diminish, vector2 texcoord, output float result)
{
    float f = mx_fbm(texcoord.x, texcoord.y, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_fractal2d_vector2.osl ---
void mx_fractal2d_vector2(vector2 amplitude, int octaves, float lacunarity, float diminish, vector2 texcoord, output vector2 result)
{
    vector2 f = mx_fbm(texcoord.x, texcoord.y, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_fractal2d_vector3.osl ---
void mx_fractal2d_vector3(vector amplitude, int octaves, float lacunarity, float diminish, vector2 texcoord, output vector result)
{
    vector f = mx_fbm(texcoord.x, texcoord.y, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_fractal2d_vector4.osl ---
void mx_fractal2d_vector4(vector4 amplitude, int octaves, float lacunarity, float diminish, vector2 texcoord, output vector4 result)
{
    vector4 f = mx_fbm(texcoord.x, texcoord.y, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_fractal3d_float.osl ---
void mx_fractal3d_float(float amplitude, int octaves, float lacunarity, float diminish, vector position, output float result)
{
    float f = mx_fbm(position, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_fractal3d_vector2.osl ---
void mx_fractal3d_vector2(vector2 amplitude, int octaves, float lacunarity, float diminish, vector position, output vector2 result)
{
    vector2 f = mx_fbm(position, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_fractal3d_vector3.osl ---
void mx_fractal3d_vector3(vector amplitude, int octaves, float lacunarity, float diminish, vector position, output vector result)
{
    vector f = mx_fbm(position, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_fractal3d_vector4.osl ---
void mx_fractal3d_vector4(vector4 amplitude, int octaves, float lacunarity, float diminish, vector position, output vector4 result)
{
    vector4 f = mx_fbm(position, octaves, lacunarity, diminish, "snoise");
    result = f * amplitude;
}

// --- mx_frame_float.osl ---
void mx_frame_float(output float result)
{
    // Use the standard default value if the attribute is not present.
    result = 1.0;
    getattribute("frame", result);
}

// --- mx_geomcolor_color3.osl ---
void mx_geomcolor_color3(int index, output color result)
{
    getattribute("color", result);
}

// --- mx_geomcolor_color4.osl ---
void mx_geomcolor_color4(int index, output color4 result)
{
    float value[4];
    getattribute("color", value);
    result.rgb[0] = value[0];
    result.rgb[1] = value[1];
    result.rgb[2] = value[2];
    result.a = value[3];
}

// --- mx_geomcolor_float.osl ---
void mx_geomcolor_float(int index, output float result)
{
    getattribute("color", result);
}

// --- mx_geompropvalue_boolean.osl ---
void mx_geompropvalue_boolean(string geomprop, int defaultVal, output int out)
{
    if (getattribute(geomprop, out) == 0)
        out = defaultVal;
}

// --- mx_geompropvalue_color3.osl ---
void mx_geompropvalue_color(string geomprop, color defaultVal, output color out)
{
    if (getattribute(geomprop, out) == 0)
        out = defaultVal;
}

// --- mx_geompropvalue_color4.osl ---
void mx_geompropvalue_color4(string geomprop, color4 defaultVal, output color4 out)
{
    float value[4];
    if (getattribute(geomprop, value) == 0)
    {
        out.rgb = defaultVal.rgb;
        out.a = defaultVal.a;
    }
    else
    {
        out.rgb[0] = value[0];
        out.rgb[1] = value[1];
        out.rgb[2] = value[2];
        out.a = value[3];
    }
}

// --- mx_geompropvalue_float.osl ---
void mx_geompropvalue_float(string geomprop, float defaultVal, output float result)
{
    if (getattribute(geomprop, result) == 0)
    {
        result = defaultVal;
    }
}

// --- mx_geompropvalue_integer.osl ---
void mx_geompropvalue_integer(string geomprop, int defaultVal, output int out)
{
    if (getattribute(geomprop, out) == 0)
        out = defaultVal;
}

// --- mx_geompropvalue_string.osl ---
void mx_geompropvalue_string(string geomprop, string defaultVal, output string out)
{
    if (getattribute(geomprop, out) == 0)
        out = defaultVal;
}

// --- mx_geompropvalue_vector2.osl ---
void mx_geompropvalue_vector2(string geomprop, vector2 defaultVal, output vector2 out)
{
    float value[2];
    if (getattribute(geomprop, value) == 0)
    {
        out = defaultVal;
    }
    else
    {
        out.x = value[0];
        out.y = value[1];
    }
}

// --- mx_geompropvalue_vector3.osl ---
void mx_geompropvalue_vector(string geomprop, vector defaultVal, output vector out)
{
    if (getattribute(geomprop, out) == 0)
        out = defaultVal;
}

// --- mx_geompropvalue_vector4.osl ---
void mx_geompropvalue_vector4(string geomprop, vector4 defaultVal, output vector4 out)
{
    float value[4];
    if (getattribute(geomprop, value) == 0)
    {
        out = defaultVal;
    }
    else
    {
        out.x = value[0];
        out.y = value[1];
        out.z = value[2];
        out.w = value[3];
    }
}

// --- mx_heighttonormal_vector3.osl ---
void mx_heighttonormal_vector3(float height, float scale, vector2 texcoord, output vector result)
{
    // Scale factor for parity with traditional Sobel filtering.
    float SOBEL_SCALE_FACTOR = 1.0 / 16.0;

    // Compute screen-space gradients of the heightfield and texture coordinates.
    vector2 dHdS = vector2(Dx(height), Dy(height)) * scale * SOBEL_SCALE_FACTOR;
    vector2 dUdS = vector2(Dx(texcoord.x), Dy(texcoord.x));
    vector2 dVdS = vector2(Dx(texcoord.y), Dy(texcoord.y));

    // Construct a screen-space tangent frame.
    vector tangent = vector(dUdS.x, dVdS.x, dHdS.x);
    vector bitangent = vector(dUdS.y, dVdS.y, dHdS.y);
    vector n = cross(tangent, bitangent);

    // Handle invalid and mirrored texture coordinates.
    if (dot(n, n) < M_FLOAT_EPS * M_FLOAT_EPS)
    {
        n = vector(0, 0, 1);
    }
    else if (n[2] < 0.0)
    {
        n *= -1.0;
    }

    // Normalize and encode the results.
    result = normalize(n) * 0.5 + 0.5;
}

// --- mx_hsvtorgb_color3.osl ---
void mx_hsvtorgb_color3(vector _in, output vector result)
{
    result = transformc("hsv","rgb", _in);
}

// --- mx_hsvtorgb_color4.osl ---
void mx_hsvtorgb_color4(color4 _in, output color4 result)
{
    result = color4(transformc("hsv","rgb", _in.rgb), 1.0);
}

// --- mx_image_color3.osl ---

void mx_image_color3(textureresource file, string layer, color default_value, vector2 texcoord, string uaddressmode, string vaddressmode, string filtertype, string framerange, int frameoffset, string frameendaction, output color out)
{
    if (file.filename == "" ||
        (uaddressmode == "constant" && (texcoord.x<0.0 || texcoord.x>1.0)) ||
        (vaddressmode == "constant" && (texcoord.y<0.0 || texcoord.y>1.0)))
    {
        out = default_value;
        return;
    }

    color missingColor = default_value;
    vector2 st = mx_transform_uv(texcoord);
    out = texture(file.filename, st.x, st.y,
                  "subimage", layer, "interp", filtertype,
                  "missingcolor", missingColor,
                  "swrap", uaddressmode, "twrap", vaddressmode
#if OSL_VERSION_MAJOR >= 1 && OSL_VERSION_MINOR >= 14
                  , "colorspace", file.colorspace
#endif
                  );
}

// --- mx_image_color4.osl ---

void mx_image_color4(textureresource file, string layer, color4 default_value, vector2 texcoord, string uaddressmode, string vaddressmode, string filtertype, string framerange, int frameoffset, string frameendaction, output color4 out)
{
    if (file.filename == "" ||
        (uaddressmode == "constant" && (texcoord.x<0.0 || texcoord.x>1.0)) ||
        (vaddressmode == "constant" && (texcoord.y<0.0 || texcoord.y>1.0)))
    {
        out = default_value;
        return;
    }

    color missingColor = default_value.rgb;
    float missingAlpha = default_value.a;
    vector2 st = mx_transform_uv(texcoord);
    float alpha;
    color rgb = texture(file.filename, st.x, st.y, "alpha", alpha,
                        "subimage", layer, "interp", filtertype,
                        "missingcolor", missingColor, "missingalpha", missingAlpha,
                        "swrap", uaddressmode, "twrap", vaddressmode
#if OSL_VERSION_MAJOR >= 1 && OSL_VERSION_MINOR >= 14
                        , "colorspace", file.colorspace
#endif
                        );

    out = color4(rgb, alpha);
}

// --- mx_image_float.osl ---

void mx_image_float(textureresource file, string layer, float default_value, vector2 texcoord, string uaddressmode, string vaddressmode, string filtertype, string framerange, int frameoffset, string frameendaction, output float out)
{
    if (file.filename == "" ||
        (uaddressmode == "constant" && (texcoord.x<0.0 || texcoord.x>1.0)) ||
        (vaddressmode == "constant" && (texcoord.y<0.0 || texcoord.y>1.0)))
    {
        out = default_value;
        return;
    }

    color missingColor = color(default_value);
    vector2 st = mx_transform_uv(texcoord);
    color rgb = texture(file.filename, st.x, st.y,
                        "subimage", layer, "interp", filtertype,
                        "missingcolor", missingColor,
                        "swrap", uaddressmode, "twrap", vaddressmode);
    out = rgb[0];
}

// --- mx_image_vector2.osl ---

void mx_image_vector2(textureresource file, string layer, vector2 default_value, vector2 texcoord, string uaddressmode, string vaddressmode, string filtertype, string framerange, int frameoffset, string frameendaction, output vector2 out)
{
    if (file.filename == "" ||
        (uaddressmode == "constant" && (texcoord.x<0.0 || texcoord.x>1.0)) ||
        (vaddressmode == "constant" && (texcoord.y<0.0 || texcoord.y>1.0)))
    {
        out = default_value;
        return;
    }

    color missingColor = color(default_value.x, default_value.y, 0.0);
    vector2 st = mx_transform_uv(texcoord);
    color rgb = texture(file.filename, st.x, st.y,
                        "subimage", layer, "interp", filtertype,
                        "missingcolor", missingColor,
                        "swrap", uaddressmode, "twrap", vaddressmode);
    out.x = rgb[0];
    out.y = rgb[1];
}

// --- mx_image_vector3.osl ---

void mx_image_vector3(textureresource file, string layer, vector default_value, vector2 texcoord, string uaddressmode, string vaddressmode, string filtertype, string framerange, int frameoffset, string frameendaction, output vector out)
{
    if (file.filename == "" ||
        (uaddressmode == "constant" && (texcoord.x<0.0 || texcoord.x>1.0)) ||
        (vaddressmode == "constant" && (texcoord.y<0.0 || texcoord.y>1.0)))
    {
        out = default_value;
        return;
    }

    color missingColor = default_value;
    vector2 st = mx_transform_uv(texcoord);
    out = texture(file.filename, st.x, st.y,
                  "subimage", layer, "interp", filtertype,
                  "missingcolor", missingColor,
                  "swrap", uaddressmode, "twrap", vaddressmode);
}

// --- mx_image_vector4.osl ---

void mx_image_vector4(textureresource file, string layer, vector4 default_value, vector2 texcoord, string uaddressmode, string vaddressmode, string filtertype, string framerange, int frameoffset, string frameendaction, output vector4 out)
{
    if (file.filename == "" ||
        (uaddressmode == "constant" && (texcoord.x<0.0 || texcoord.x>1.0)) ||
        (vaddressmode == "constant" && (texcoord.y<0.0 || texcoord.y>1.0)))
    {
        out = default_value;
        return;
    }

    color missingColor = color(default_value.x, default_value.y, default_value.z);
    float missingAlpha = default_value.w;
    vector2 st = mx_transform_uv(texcoord);
    float alpha;
    color rgb = texture(file.filename, st.x, st.y, "alpha", alpha,
                        "subimage", layer, "interp", filtertype,
                        "missingcolor", missingColor, "missingalpha", missingAlpha,
                        "swrap", uaddressmode, "twrap", vaddressmode);

    out = vector4(rgb[0], rgb[1], rgb[2], alpha);
}

// --- mx_luminance_color3.osl ---
void mx_luminance_color3(color in, color lumacoeffs, output color result)
{
    result = dot(in, lumacoeffs);
}

// --- mx_luminance_color4.osl ---
void mx_luminance_color4(color4 in, color lumacoeffs, output color4 result)
{
    result = color4(dot(in.rgb, lumacoeffs), in.a);
}

// --- mx_mix_surfaceshader.osl ---
void mx_mix_surfaceshader(surfaceshader fg, surfaceshader bg, float w, output surfaceshader result)
{
    result.bsdf = mix(bg.bsdf, fg.bsdf, w);
    result.edf = mix(bg.edf, fg.edf, w);
    result.opacity = mix(bg.opacity, fg.opacity, w);
}

// --- mx_noise2d_float.osl ---
void mx_noise2d_float(float amplitude, float pivot, vector2 texcoord, output float result)
{
    float value = noise("snoise", texcoord.x, texcoord.y);
    result = value * amplitude + pivot;
}

// --- mx_noise2d_vector2.osl ---
void mx_noise2d_vector2(vector2 amplitude, float pivot, vector2 texcoord, output vector2 result)
{
    vector2 value = mx_noise("snoise", texcoord.x, texcoord.y);
    result = value * amplitude + pivot;
}

// --- mx_noise2d_vector3.osl ---
void mx_noise2d_vector3(vector amplitude, float pivot, vector2 texcoord, output vector result)
{
    vector value = noise("snoise", texcoord.x, texcoord.y);
    result = value * amplitude + pivot;
}

// --- mx_noise2d_vector4.osl ---
void mx_noise2d_vector4(vector4 amplitude, float pivot, vector2 texcoord, output vector4 result)
{
    vector4 value = mx_noise("snoise", texcoord.x, texcoord.y);
    result = value * amplitude + pivot;
}

// --- mx_noise3d_float.osl ---
void mx_noise3d_float(float amplitude, float pivot, vector position, output float result)
{
    float value = noise("snoise", position);
    result = value * amplitude + pivot;
}

// --- mx_noise3d_vector2.osl ---
void mx_noise3d_vector2(vector2 amplitude, float pivot, vector position, output vector2 result)
{
    vector2 value = mx_noise("snoise", position);
    result = value * amplitude + pivot;
}

// --- mx_noise3d_vector3.osl ---
void mx_noise3d_vector3(vector amplitude, float pivot, vector position, output vector result)
{
    vector value = noise("snoise", position);
    result = value * amplitude + pivot;
}

// --- mx_noise3d_vector4.osl ---
void mx_noise3d_vector4(vector4 amplitude, float pivot, vector position, output vector4 result)
{
    vector4 value = mx_noise("snoise", position);
    result = value * amplitude + pivot;
}

// --- mx_normalmap.osl ---
void mx_normalmap_vector2(vector value, vector2 normal_scale, vector N, vector T, vector B, output vector result)
{
    if (value == vector(0.0))
    {
        result = N;
    }
    else
    {
        // The OSL backend uses dPdu and dPdv for tangents and bitangents, but these vectors are not
        // guaranteed to be orthonormal.
        //
        // Orthogonalize the tangent frame using Gram-Schmidt, unlike in the other backends.
        //
        vector v = value * 2.0 - 1.0;
        vector Tn = normalize(T - dot(T, N) * N);
        vector Bn = normalize(B - dot(B, N) * N - dot(B, Tn) * Tn);
        result = normalize(Tn * v[0] * normal_scale.x + Bn * v[1] * normal_scale.y + N * v[2]);
    }
}

void mx_normalmap_float(vector value, float normal_scale, vector N, vector T, vector B, output vector result)
{
    mx_normalmap_vector2(value, vector2(normal_scale, normal_scale), N, T, B, result);
}

// --- NG_checkerboard_color3 ---
// Expanded inline from MaterialX stdlib_ng.mtlx NG_checkerboard_color3.
// Inputs match ND_checkerboard_color3: color1, color2, uvtiling, uvoffset, texcoord.
void NG_checkerboard_color3(color color1, color color2,
                             vector2 uvtiling, vector2 uvoffset,
                             vector2 texcoord, output color out)
{
    float u = texcoord[0] * uvtiling[0] + uvoffset[0];
    float v = texcoord[1] * uvtiling[1] + uvoffset[1];
    float checker = mod(floor(u) + floor(v), 2.0);
    out = mix(color1, color2, checker);
}

// --- mx_premult_color4.osl ---
void mx_premult_color4(color4 in, output color4 result)
{
    result = color4(in.rgb * in.a, in.a);
}

// --- mx_rgbtohsv_color3.osl ---
void mx_rgbtohsv_color3(vector _in, output vector result)
{
    result = transformc("rgb","hsv", _in);
}

// --- mx_rgbtohsv_color4.osl ---
void mx_rgbtohsv_color4(color4 _in, output color4 result)
{
    result = color4(transformc("rgb","hsv", _in.rgb), 1.0);
}

// --- mx_rotate_vector2.osl ---
void mx_rotate_vector2(vector2 _in, float amount, output vector2 result)
{
    float rotationRadians = radians(amount);
    float sa = sin(rotationRadians);
    float ca = cos(rotationRadians);
    result = vector2(ca*_in.x + sa*_in.y, -sa*_in.x + ca*_in.y);
}

// --- mx_rotate_vector3.osl ---
matrix rotationMatrix(vector axis, float angle)
{
    vector nAxis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return matrix(oc * nAxis[0] * nAxis[0] + c,             oc * nAxis[0] * nAxis[1] - nAxis[2] * s,  oc * nAxis[2] * nAxis[0] + nAxis[1] * s,  0.0,
                  oc * nAxis[0] * nAxis[1] + nAxis[2] * s,  oc * nAxis[1] * nAxis[1] + c,             oc * nAxis[1] * nAxis[2] - nAxis[0] * s,  0.0,
                  oc * nAxis[2] * nAxis[0] - nAxis[1] * s,  oc * nAxis[1] * nAxis[2] + nAxis[0] * s,  oc * nAxis[2] * nAxis[2] + c,             0.0,
                  0.0,                                      0.0,                                      0.0,                                      1.0);
}

void mx_rotate_vector3(vector _in, float amount, vector axis, output vector result)
{
    float rotationRadians = radians(amount);
    matrix m = rotationMatrix(axis, rotationRadians);
    vector4 trans = transform(m, vector4(_in[0], _in[1], _in[2], 1.0));
    result = vector(trans.x, trans.y, trans.z);
}

// --- mx_surface_unlit.osl ---
void mx_surface_unlit(float emission_weight, color emission_color, float transmission_weight, color transmission_color, float opacity, output surfaceshader result)
{
    float trans = clamp(transmission_weight, 0.0, 1.0);
    result.bsdf = trans * transmission_color * transparent();
    result.edf  = (1.0 - trans) * emission_weight * emission_color * emission();
    result.opacity = clamp(opacity, 0.0, 1.0);
}

// --- mx_time_float.osl ---
void mx_time_float(float fps, output float result)
{
    // Use the standard default value if the attribute is not present.
    result = 0.0;
    getattribute("time", result);
}

// --- mx_transformmatrix_vector2M3.osl ---
void mx_transformmatrix_vector2M3(vector2 val, matrix m, output vector2 result)
{
    point res = transform(m, point(val.x, val.y, 1.0));
    result.x = res[0];
    result.y = res[1];
}

// --- mx_unpremult_color4.osl ---
void mx_unpremult_color4(color4 in, output color4 result)
{
    result = color4(in.rgb / in.a, in.a);
}

// --- mx_worleynoise2d_float.osl ---
void mx_worleynoise2d_float(vector2 texcoord, float jitter, int style, output float result)
{
    result = mx_worley_noise_float(texcoord, jitter, style, 0);
}

// --- mx_worleynoise2d_vector2.osl ---
void mx_worleynoise2d_vector2(vector2 texcoord, float jitter, int style, output vector2 result)
{
    result = mx_worley_noise_vector2(texcoord, jitter, style, 0);
}

// --- mx_worleynoise2d_vector3.osl ---
void mx_worleynoise2d_vector3(vector2 texcoord, float jitter, int style, output vector result)
{
    result = mx_worley_noise_vector3(texcoord, jitter, style, 0);
}

// --- mx_worleynoise3d_float.osl ---
void mx_worleynoise3d_float(vector position, float jitter, int style, output float result)
{
    result = mx_worley_noise_float(position, jitter, style, 0);
}

// --- mx_worleynoise3d_vector2.osl ---
void mx_worleynoise3d_vector2(vector position, float jitter, int style, output vector2 result)
{
    result = mx_worley_noise_vector2(position, jitter, style, 0);
}

// --- mx_worleynoise3d_vector3.osl ---
void mx_worleynoise3d_vector3(vector position, float jitter, int style, output vector result)
{
    result = mx_worley_noise_vector3(position, jitter, style, 0);
}


// ---- NG_place2d_vector2 (from stdlib_ng.mtlx, no genosl counterpart) ----
vector2 mx_rotate2d(vector2 v, float degrees) {
    float r = radians(degrees);
    float c = cos(r), s = sin(r);
    return vector2(v.x*c - v.y*s, v.x*s + v.y*c);
}
void NG_place2d_vector2(
    vector2 texcoord, vector2 pivot, vector2 scale,
    float rotate, vector2 offset, int operationorder,
    output vector2 result)
{
    vector2 p = texcoord - pivot;
    if (operationorder == 0) {
        // SRT: scale -> rotate -> translate
        float sx = (scale.x != 0.0) ? p.x / scale.x : p.x;
        float sy = (scale.y != 0.0) ? p.y / scale.y : p.y;
        vector2 ro = mx_rotate2d(vector2(sx, sy), rotate);
        result = ro - offset + pivot;
    } else {
        // TRS: translate -> rotate -> scale
        vector2 tr = p - offset;
        vector2 ro = mx_rotate2d(tr, rotate);
        float rx = (scale.x != 0.0) ? ro.x / scale.x : ro.x;
        float ry = (scale.y != 0.0) ? ro.y / scale.y : ro.y;
        result = vector2(rx, ry) + pivot;
    }
}
