from pyproj import Transformer

# Coordinate transformations
transformer_4326_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857")
transformer_3857_to_4326 = Transformer.from_crs("epsg:3857", "epsg:4326")


# epsg4326: Lon/lat
# epsg3857: Easting/Northing (Spherical Mercator)
def ll2en(coords):  # epsg_4326_to_3857
    return [transformer_4326_to_3857.transform(*reversed(row)) for row in coords]


def en2ll(coords):  # epsg_3857_to_4326
    return [list(reversed(transformer_3857_to_4326.transform(*row))) for row in coords]


scheduler_url = "127.0.0.1:8786"
