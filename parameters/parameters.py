# PARAMETERS

SERVER = False # Whether the code runs on the server (important to avoid parallel job conflicts)


networktypes = ["biketrack", "carall", "bikeable", "biketrackcarall", "biketrack_onstreet", "bikeable_offstreet"] # Existing infrastructures to analyze

gridl = 300 # in m, for generating the grid
bearingbins = 72 # number of bins to determine bearing. e.g. 72 will create 5 degrees bins
buffer_walk = 500 # Buffer in m for coverage calculations. (How far people are willing to walk)
buffer_accident = 50 # Buffer in m for crashes coverage calculations.
numnodepairs = 500 # Number of node pairs to consider for random sample to calculate trips_coverage,directness 
detour = 0 # fraction of detour for trip coverage metric. It must be in [0,1). E.g. detour = 0.25 corresponds to 25% detour

# font sizes for plots
axis_size = 5.5 # axis label font size
title_size = 6.5 # title font size
ticks_size = 5.5 # ticks font size
label_size = 4.5 # legend font size

line_widths = 0.7 # line width in plots
marker_size = 4 # marker size in plots

nodesize_grown = 7.5
plotparam = {"bbox": (1280,1280),
			"dpi": 300,
			"carall_black": {"width": .2, "edge_color": 'black','alpha':0.9},
			"carall": {"width": .2, "edge_color": '#999999'},
			"carall_tiny": {"width": 0.18, "edge_color": '#999999'},
			"carall_tiny_red": {"width": 0.05, "edge_color": 'red'},
			"bikegrown_new": {"width": 1.3, "edge_color": 'navy', "label": "New Infrastructure"},
			"bikegrown_new0": {"width": 1.1, "edge_color": '#ec2424', "label": "New Infrastructure"},
			"bikegrown_new1": {"width": 1.1, "edge_color": 'navy', "label": "New Infrastructure"},
			"bikegrown_existing": {"width": .7, "edge_color": '#00b15c', "label": "Existing Infrastructure","alpha":0.5},
			"bikegrown_existing_highlight": {"width": 1., "edge_color": '#00b15c', "label": "Existing Infrastructure","alpha":0.5},
			"poi_unreached": {"node_color": '#ffa600', "edgecolors": 'navy','linewidths':0.1},
			"abstract": {"width": 1.4,"edge_color": "navy", "alpha": 0.7}
            }

# CONSTANTS
# These values should be set once and not be changed

osmnxparameters = {'car30': {'network_type':'drive', 'custom_filter':'["maxspeed"~"^30$|^20$|^15$|^10$|^5$|^20 mph|^15 mph|^10 mph|^5 mph"]', 'export': True, 'retain_all': True},
                   'carall': {'network_type':'drive', 'custom_filter': None, 'export': True, 'retain_all': False},
                   'bike_cyclewaytrack': {'network_type':'bike', 'custom_filter':'["cycleway"~"track"]', 'export': False, 'retain_all': True},
                   'bike_highwaycycleway': {'network_type':'bike', 'custom_filter':'["highway"~"cycleway"]', 'export': False, 'retain_all': True},
                   'bike_designatedpath': {'network_type':'all', 'custom_filter':'["highway"~"path"]["bicycle"~"designated"]', 'export': False, 'retain_all': True},
                   'bike_cyclewayrighttrack': {'network_type':'bike', 'custom_filter':'["cycleway:right"~"track"]', 'export': False, 'retain_all': True},
                   'bike_cyclewaylefttrack': {'network_type':'bike', 'custom_filter':'["cycleway:left"~"track"]', 'export': False, 'retain_all': True},
                   'bike_cyclestreet': {'network_type':'bike', 'custom_filter':'["cyclestreet"]', 'export': False, 'retain_all': True},
                   'bike_bicycleroad': {'network_type':'bike', 'custom_filter':'["bicycle_road"]', 'export': False, 'retain_all': True},
                   'bike_livingstreet': {'network_type':'bike', 'custom_filter':'["highway"~"living_street"]', 'export': False, 'retain_all': True}
                  }  
# Special case 'biketrack': "cycleway"~"track" OR "highway"~"cycleway" OR "bicycle"~"designated" OR "cycleway:right=track" OR "cycleway:left=track" OR ("highway"~"path" AND "bicycle"~"designated") OR "cyclestreet" OR "highway"~"living_street"
# Special case 'bikeable': biketrack OR car30
# See: https://wiki.openstreetmap.org/wiki/Key:cycleway#Cycle_tracks
# https://wiki.openstreetmap.org/wiki/Tag:highway=path#Usage_as_a_universal_tag
# https://wiki.openstreetmap.org/wiki/Tag:highway%3Dliving_street
# https://wiki.openstreetmap.org/wiki/Key:cyclestreet


# 02
snapthreshold = 700 # in m, tolerance for snapping POIs to network

print("Loaded parameters.\n")
