"""Time-of-day / weather presets for the Forest Valley scene.

Self-contained (bpy only) so the same file is embedded in the .blend as a Text block:
open it in Blender's Text Editor, set PRESET at the bottom and Run Script to switch.

Nodes and objects are found by the fixed names created by build_lighting.py:
  world nodes  ATM_Sky, ATM_SkyStrength, ATM_Stars, ATM_CloudsWorld
  objects      ATM_Sun, ATM_Clouds, ATM_Fog, ATM_Rain, ATM_Night_Lights (collection)
  materials    M_ATM_Clouds (ATM_CloudCoverage, ATM_CloudDark), M_ATM_Fog (ATM_FogDensity, ATM_FogHeight, ATM_FogColor)
  scene props  wetness (read by terrain / road / rock materials through a View Layer attribute)
"""
import math

import bpy

PRESETS = {
    # sun: elevation / azimuth (deg, azimuth clockwise from north), irradiance (Blender W/m2), colour
    "morning": {"sun_elev": 9.0, "sun_az": 75.0, "sun_energy": 4.0, "sun_color": (1.0, 0.80, 0.62), "sun_angle": 1.2,
                "sky_strength": 0.42, "air": 1.0, "aerosol": 2.2, "stars": 0.0, "exposure": 0.3,
                "fog_density": 0.006, "fog_height": 9.0, "fog_color": (0.92, 0.88, 0.82),
                "cloud_coverage": 0.38, "cloud_dark": 0.15, "rain": False, "wetness": 0.25, "night_lights": False,
                "ue": {"color_temperature_k": 4300, "sun_lux": 20000, "height_fog_density": 0.06,
                       "height_fog_falloff": 0.35, "volumetric_fog": True, "cloud_coverage": 0.38}},
    "noon": {"sun_elev": 58.0, "sun_az": 165.0, "sun_energy": 6.0, "sun_color": (1.0, 0.97, 0.93), "sun_angle": 0.6,
             "sky_strength": 0.22, "air": 1.0, "aerosol": 1.0, "stars": 0.0, "exposure": -0.1,
             "fog_density": 0.00035, "fog_height": 40.0, "fog_color": (0.75, 0.82, 0.95),
             "cloud_coverage": 0.2, "cloud_dark": 0.1, "rain": False, "wetness": 0.0, "night_lights": False,
             "ue": {"color_temperature_k": 5800, "sun_lux": 100000, "height_fog_density": 0.01,
                    "height_fog_falloff": 0.15, "volumetric_fog": False, "cloud_coverage": 0.32}},
    "sunset": {"sun_elev": 6.0, "sun_az": 285.0, "sun_energy": 5.0, "sun_color": (1.0, 0.5, 0.22), "sun_angle": 1.5,
               "sky_strength": 0.55, "air": 1.2, "aerosol": 3.0, "stars": 0.0, "exposure": 0.6,
               "fog_density": 0.0015, "fog_height": 18.0, "fog_color": (1.0, 0.75, 0.55),
               "cloud_coverage": 0.45, "cloud_dark": 0.25, "rain": False, "wetness": 0.0, "night_lights": False,
               "ue": {"color_temperature_k": 2800, "sun_lux": 4000, "height_fog_density": 0.03,
                      "height_fog_falloff": 0.25, "volumetric_fog": True, "cloud_coverage": 0.45}},
    "night": {"sun_elev": 38.0, "sun_az": 210.0, "sun_energy": 0.06, "sun_color": (0.62, 0.72, 1.0), "sun_angle": 0.5,
              "sky_strength": 0.0, "air": 1.0, "aerosol": 1.0, "stars": 1.0, "exposure": 3.2,
              "fog_density": 0.003, "fog_height": 15.0, "fog_color": (0.6, 0.68, 0.85),
              "cloud_coverage": 0.25, "cloud_dark": 0.85, "rain": False, "wetness": 0.15, "night_lights": True,
              "ue": {"color_temperature_k": 4100, "moon_lux": 0.3, "height_fog_density": 0.02,
                     "height_fog_falloff": 0.3, "volumetric_fog": True, "cloud_coverage": 0.25}},
    "foggy": {"sun_elev": 24.0, "sun_az": 120.0, "sun_energy": 0.9, "sun_color": (0.95, 0.95, 0.95), "sun_angle": 8.0,
              "sky_strength": 0.3, "air": 1.0, "aerosol": 6.0, "stars": 0.0, "exposure": 0.3,
              "fog_density": 0.007, "fog_height": 35.0, "fog_color": (0.85, 0.87, 0.9),
              "cloud_coverage": 0.95, "cloud_dark": 0.25, "rain": False, "wetness": 0.35, "night_lights": False,
              "ue": {"color_temperature_k": 6500, "sun_lux": 8000, "height_fog_density": 0.35,
                     "height_fog_falloff": 0.05, "volumetric_fog": True, "cloud_coverage": 0.95}},
    "rainy": {"sun_elev": 30.0, "sun_az": 140.0, "sun_energy": 0.5, "sun_color": (0.85, 0.88, 0.95), "sun_angle": 10.0,
              "sky_strength": 0.22, "air": 1.0, "aerosol": 5.0, "stars": 0.0, "exposure": 0.4,
              "fog_density": 0.004, "fog_height": 30.0, "fog_color": (0.6, 0.64, 0.7),
              "cloud_coverage": 1.0, "cloud_dark": 0.6, "rain": True, "wetness": 1.0, "night_lights": False,
              "ue": {"color_temperature_k": 7000, "sun_lux": 5000, "height_fog_density": 0.12,
                     "height_fog_falloff": 0.08, "volumetric_fog": True, "cloud_coverage": 1.0,
                     "rain_spawn_per_s": 40000, "material_wetness": 1.0}},
}


def sun_rotation(elev_deg, az_deg):
    """Blender sun Euler for a sun at elevation / azimuth (clockwise from +Y north)."""
    return (math.radians(90.0 - elev_deg), 0.0, math.radians(180.0 - az_deg))


def apply(scene, name):
    p = PRESETS[name]
    w = scene.world
    N = w.node_tree.nodes
    sky = N.get("ATM_Sky")
    sky.sun_elevation = math.radians(p["sun_elev"])
    sky.sun_rotation = math.radians(p["sun_az"])
    sky.air_density = p["air"]
    sky.aerosol_density = p["aerosol"]
    N["ATM_SkyStrength"].inputs[1].default_value = p["sky_strength"]
    N["ATM_Stars"].inputs[1].default_value = p["stars"]
    N["ATM_CloudsWorld"].inputs[0].default_value = p["cloud_coverage"]
    sun = bpy.data.objects["ATM_Sun"]
    sun.rotation_euler = sun_rotation(p["sun_elev"], p["sun_az"])
    sun.data.energy = p["sun_energy"]
    sun.data.color = p["sun_color"]
    sun.data.angle = math.radians(p["sun_angle"])
    cm = bpy.data.materials["M_ATM_Clouds"].node_tree.nodes
    cm["ATM_CloudCoverage"].outputs[0].default_value = p["cloud_coverage"]
    cm["ATM_CloudDark"].outputs[0].default_value = p["cloud_dark"]
    fm = bpy.data.materials["M_ATM_Fog"].node_tree.nodes
    fm["ATM_FogDensity"].outputs[0].default_value = p["fog_density"]
    fm["ATM_FogHeight"].outputs[0].default_value = p["fog_height"]
    fm["ATM_FogColor"].outputs[0].default_value = (*p["fog_color"], 1.0)
    rain = bpy.data.objects.get("ATM_Rain")
    if rain:
        rain.hide_render = rain.hide_viewport = not p["rain"]
    nl = bpy.data.collections.get("ATM_Night_Lights")
    if nl:
        nl.hide_render = nl.hide_viewport = not p["night_lights"]
    scene["wetness"] = p["wetness"]
    scene.view_settings.exposure = p["exposure"]
    scene["lighting_preset"] = name


if __name__ == "__main__":
    PRESET = "noon"          # morning / noon / sunset / night / foggy / rainy
    apply(bpy.context.scene, PRESET)
