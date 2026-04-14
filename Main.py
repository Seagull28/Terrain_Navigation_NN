from src.TerrainNavigator import Navigator

datapath = "data/TRN/"

navigator = Navigator(
    2000,
    "ReferenceMap.ppm",
    "catalogue",
    datapath
)

navigator.locateDescentImageInReferenceImage(datapath + "Angle1.ppm")