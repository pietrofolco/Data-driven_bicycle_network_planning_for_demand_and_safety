def holepatchlist_from_cov(cov, map_center):
    """Get a patchlist of holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holeseq_per_poly = get_holes(cov)
    holepatchlist = []
    for hole_per_poly in holeseq_per_poly:
        for hole in hole_per_poly:
            holepatchlist.append(hole_to_patch(hole, map_center))
    return holepatchlist

def fill_holes(cov):
    """Fill holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holeseq_per_poly = get_holes(cov)
    holes = []
    for hole_per_poly in holeseq_per_poly:
        for hole in hole_per_poly:
            holes.append(hole)
    eps = 0.00000001
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        cov_filled = ops.unary_union([poly for poly in cov] + [Polygon(hole).buffer(eps) for hole in holes])
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        cov_filled = ops.unary_union([cov] + [Polygon(hole).buffer(eps) for hole in holes])
    return cov_filled

def extract_relevant_polygon(placeid, mp):
    """Return the most relevant polygon of a multipolygon mp, for being considered the city limit.
    Depends on location.
    """
    if isinstance(mp, shapely.geometry.polygon.Polygon):
        return mp
    if placeid == "tokyo": # If Tokyo, take poly with most northern bound, otherwise largest
        p = max(mp, key=lambda a: a.bounds[-1])
    else:
        p = max(mp, key=lambda a: a.area)
    return p

def get_holes(cov):
    """Get holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holes = []
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        for pol in cov: # cov is generally a MultiPolygon, so we iterate through its Polygons
            holes.append(pol.interiors)
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        holes.append(cov.interiors)
    return holes

def cov_to_patchlist(cov, map_center, return_holes = True):
    """Turns a coverage Polygon or MultiPolygon into a matplotlib patch list, for plotting
    """
    p = []
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        for pol in cov: # cov is generally a MultiPolygon, so we iterate through its Polygons
            p.append(pol_to_patch(pol, map_center))
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        p.append(pol_to_patch(cov, map_center))
    if not return_holes:
        return p
    else:
        holepatchlist = holepatchlist_from_cov(cov, map_center)
        return p, holepatchlist

def pol_to_patch(pol, map_center):
    """Turns a coverage Polygon into a matplotlib patch, for plotting
    """
    y, x = pol.exterior.coords.xy
    pos_transformed, _ = project_pos(y, x, map_center)
    return matplotlib.patches.Polygon(pos_transformed)

def hole_to_patch(hole, map_center):
    """Turns a LinearRing (hole) into a matplotlib patch, for plotting
    """
    y, x = hole.coords.xy
    pos_transformed, _ = project_pos(y, x, map_center)
    return matplotlib.patches.Polygon(pos_transformed)


def set_analysissubplot(key):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if key in ["length", "length_lcc", "coverage", "poi_coverage", "overlap_bikeable", "overlap_biketrack", "components", "efficiency_local", "efficiency_global"]:
        ax.set_ylim(bottom = 0)
    if key in ["directness_lcc"]:
        ax.set_ylim(bottom = 0.2)
    if key in ["directness_lcc", "efficiency_global", "efficiency_local"]:
        ax.set_ylim(top = 1)


def initplot():
    fig = plt.figure(figsize=(plotparam["bbox"][0]/plotparam["dpi"], plotparam["bbox"][1]/plotparam["dpi"]), dpi=plotparam["dpi"])
    plt.axes().set_aspect('equal')
    plt.axes().set_xmargin(0.01)
    plt.axes().set_ymargin(0.01)
    plt.axes().set_axis_off()
    return fig

def nodesize_from_pois(nnids):
    """Determine POI node size based on number of POIs.
    The more POIs the smaller (linearly) to avoid overlaps.
    """
    minnodesize = 30
    maxnodesize = 220
    return max(minnodesize, maxnodesize-len(nnids))


def simplify_ig(G):
    """Simplify an igraph with ox.simplify_graph
    """
    G_temp = copy.deepcopy(G)
    G_temp.es["length"] = G_temp.es["weight"]
    output = ig.Graph.from_networkx(ox.simplify_graph(nx.MultiDiGraph(G_temp.to_networkx())).to_undirected())
    output.es["weight"] = output.es["length"]
    return output


def nxdraw(G, networktype, map_center = False, nnids = False, drawfunc = "nx.draw", nodesize = 0, weighted = False, maxwidthsquared = 0, simplified = False):
    """Take an igraph graph G and draw it with a networkx drawfunc.
    """
    
    if simplified:
        G.es["length"] = G.es["weight"]
        G_nx = ox.simplify_graph(nx.MultiDiGraph(G.to_networkx())).to_undirected()
    else:
        G_nx = G.to_networkx()
    if nnids is not False: # Restrict to nnids node ids
        nnids_nx = [k for k,v in dict(G_nx.nodes(data=True)).items() if v['id'] in nnids]
        G_nx = G_nx.subgraph(nnids_nx)
        
    pos_transformed, map_center = project_nxpos(G_nx, map_center)
    
    if weighted is True:
        # The max width should be the node diameter (=sqrt(nodesize))
        widths = [x for x in list(nx.get_edge_attributes(G_nx, "bw").values())]
        #widths = list(nx.get_edge_attributes(G_nx, "n_trips").values())
        widthfactor = 1.1 * math.sqrt(maxwidthsquared) / max(widths)
        widths = [max(0.33, w * widthfactor) for w in widths]
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize, width = widths)
    elif type(weighted) is float or type(weighted) is int and weighted > 0:
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize, width = weighted)
    else:
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize)
    return map_center

def common_entries(*dcts):
    """Like zip() but for dicts.
    See: https://stackoverflow.com/questions/16458340/python-equivalent-of-zip-for-dictionaries
    """
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def project_nxpos(G, map_center = False):
    """Take a spatial nx network G and projects its GPS coordinates to local azimuthal.
    Returns transformed positions, as used by nx.draw()
    """
    lats = nx.get_node_attributes(G, 'x')
    lons = nx.get_node_attributes(G, 'y')
    pos = {nid:(lat,-lon) for (nid,lat,lon) in common_entries(lats,lons)}
    if map_center:
        loncenter = map_center[0]
        latcenter = map_center[1]
    else:
        loncenter = np.mean(list(lats.values()))
        latcenter = -1* np.mean(list(lons.values()))
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    pos_transformed = {nid:list(ops.transform(wgs84_to_aeqd.transform, Point(latlon)).coords)[0] for nid, latlon in pos.items()}
    return pos_transformed, (loncenter,latcenter)


def project_pos(lats, lons, map_center = False):
    """Project GPS coordinates to local azimuthal.
    """
    pos = [(lat,-lon) for lat,lon in zip(lats,lons)]
    if map_center:
        loncenter = map_center[0]
        latcenter = map_center[1]
    else:
        loncenter = np.mean(list(lats.values()))
        latcenter = -1* np.mean(list(lons.values()))
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    pos_transformed = [(ops.transform(wgs84_to_aeqd.transform, Point(latlon)).coords)[0] for latlon in pos]
    return pos_transformed, (loncenter,latcenter)


def round_coordinates(G, r = 7):
    for v in G.vs:
        G.vs[v.index]["x"] = round(G.vs[v.index]["x"], r)
        G.vs[v.index]["y"] = round(G.vs[v.index]["y"], r)

def mirror_y(G):
    for v in G.vs:
        y = G.vs[v.index]["y"]
        G.vs[v.index]["y"] = -y
    
def dist(v1, v2):
    dist = haversine((v1['y'],v1['x']),(v2['y'],v2['x']), unit="m") # x is lon, y is lat
    return dist

def dist_vector(v1_list, v2_list):
    dist_list = haversine_vector(v1_list, v2_list, unit="m") # [(lat,lon)], [(lat,lon)]
    return dist_list

def osm_to_ig(node, edge):
    """ Turns a node and edge dataframe into an igraph Graph.
    """
    
    G = ig.Graph(directed = False)

    x_coords = node['x'].tolist() 
    y_coords = node['y'].tolist()
    ids = node['osmid'].tolist()
    coords = []

    for i in range(len(x_coords)):
        G.add_vertex(x = x_coords[i], y = y_coords[i], id = ids[i])
        coords.append((x_coords[i], y_coords[i]))

    id_dict = dict(zip(G.vs['id'], np.arange(0, G.vcount()).tolist()))
    coords_dict = dict(zip(np.arange(0, G.vcount()).tolist(), coords))

    edge_list = []
    edge_info = {}
    edge_info["weight"] = []
    edge_info["osmid"] = []
    for i in range(len(edge)):
        edge_list.append([id_dict.get(edge['u'][i]), id_dict.get(edge['v'][i])])
        edge_info["weight"].append(round(edge['length'][i], 10))
        edge_info["osmid"].append(edge['osmid'][i])

    G.add_edges(edge_list) # attributes = edge_info doesn't work although it should: https://igraph.org/python/doc/igraph.Graph-class.html#add_edges
    for i in range(len(edge)):
        G.es[i]["weight"] = edge_info["weight"][i]
        G.es[i]["osmid"] = edge_info["osmid"][i]

    G.simplify(combine_edges=max)
    return G


def compress_file(p, f, filetype = ".csv", delete_uncompressed = True):
    with zipfile.ZipFile(p + f + ".zip", 'w', zipfile.ZIP_DEFLATED) as zfile:
        zfile.write(p + f + filetype, f + filetype)
    if delete_uncompressed: os.remove(p + f + filetype)

def ox_to_csv(G, p, placeid, parameterid, postfix = "", compress = True, verbose = True):
    if "crs" not in G.graph:
        G.graph["crs"] = 'epsg:4326' # needed for OSMNX's graph_to_gdfs in utils_graph.py
    try:
        node, edge = ox.graph_to_gdfs(G)
    except ValueError:
        node, edge = gpd.GeoDataFrame(), gpd.GeoDataFrame()
    prefix = placeid + '_' + parameterid + postfix

    node.to_csv(p + prefix + '_nodes.csv', index = False)
    if compress: compress_file(p, prefix + '_nodes')
 
    edge.to_csv(p + prefix + '_edges.csv', index = False)
    if compress: compress_file(p, prefix + '_edges')

    if verbose: print(placeid + ": Successfully wrote graph " + parameterid + postfix)

def check_extract_zip(p, prefix):
    """ Check if a zip file prefix+'_nodes.zip' and + prefix+'_edges.zip'
    is available at path p. If so extract it and return True, otherwise False.
    If you call this function, remember to clean up (i.e. delete the unzipped files)
    after you are done like this:

    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
    """

    try: # Use zip files if available
        with zipfile.ZipFile(p + prefix + '_nodes.zip', 'r') as zfile:
            zfile.extract(prefix + '_nodes.csv', p)
        with zipfile.ZipFile(p + prefix + '_edges.zip', 'r') as zfile:
            zfile.extract(prefix + '_edges.csv', p)
        return True
    except:
        return False


def csv_to_ox(p, placeid, parameterid):
    """ Load a networkx graph from _edges.csv and _nodes.csv
    The edge file must have attributes u,v,osmid,length
    The node file must have attributes y,x,osmid
    Only these attributes are loaded.
    """
    prefix = placeid + '_' + parameterid
    compress = check_extract_zip(p, prefix)
    
    with open(p + prefix + '_edges.csv', 'r') as f:
        header = f.readline().strip().split(",")

        lines = []
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = str(eval(line_list[header.index("osmid")])[0]) if isinstance(eval(line_list[header.index("osmid")]), list) else line_list[header.index("osmid")] # If this is a list due to multiedges, just load the first osmid
            length = str(eval(line_list[header.index("length")])[0]) if isinstance(eval(line_list[header.index("length")]), list) else line_list[header.index("length")] # If this is a list due to multiedges, just load the first osmid
            line_string = "" + line_list[header.index("u")] + " "+ line_list[header.index("v")] + " " + osmid + " " + length
            lines.append(line_string)
        G = nx.parse_edgelist(lines, nodetype = int, data = (("osmid", int),("length", float)), create_using = nx.MultiDiGraph) # MultiDiGraph is necessary for OSMNX, for example for get_undirected(G) in utils_graph.py
    with open(p + prefix + '_nodes.csv', 'r') as f:
        header = f.readline().strip().split(",")
        values_x = {}
        values_y = {}
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = int(line_list[header.index("osmid")])
            values_x[osmid] = float(line_list[header.index("x")])
            values_y[osmid] = float(line_list[header.index("y")])

        nx.set_node_attributes(G, values_x, "x")
        nx.set_node_attributes(G, values_y, "y")

    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
    return G


def csv_to_ig(p, placeid, parameterid, cleanup = True):
    """ Load an ig graph from _edges.csv and _nodes.csv
    The edge file must have attributes u,v,osmid,length
    The node file must have attributes y,x,osmid
    Only these attributes are loaded.
    """
    prefix = placeid + '_' + parameterid
    compress = check_extract_zip(p, prefix)
    empty = False
    try:
        n = pd.read_csv(p + prefix + '_nodes.csv')
        e = pd.read_csv(p + prefix + '_edges.csv')
    except:
        empty = True
    if compress and cleanup and not SERVER: # do not clean up on the server as csv is needed in parallel jobs
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
    if empty:
        return ig.Graph(directed = False)
    G = osm_to_ig(n, e)
    round_coordinates(G)
    mirror_y(G)
    return G


def ig_to_geojson(G):
    linestring_list = []
    for e in G.es():
        linestring_list.append(geojson.LineString([(e.source_vertex["x"], -e.source_vertex["y"]), (e.target_vertex["x"], -e.target_vertex["y"])]))
    G_geojson = geojson.GeometryCollection(linestring_list)
    return G_geojson



def clusterindices_by_length(clusterinfo, rev = True):
    return [k for k, v in sorted(clusterinfo.items(), key=lambda item: item[1]["length"], reverse = rev)]

class MyPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def segments_intersect(A,B,C,D):
    """Check if two line segments intersect (except for colinearity)
    Returns true if line segments AB and CD intersect properly.
    Adapted from: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    """
    if (A.x == C.x and A.y == C.y) or (A.x == D.x and A.y == D.y) or (B.x == C.x and B.y == C.y) or (B.x == D.x and B.y == D.y): return False # If the segments share an endpoint they do not intersect properly
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def new_edge_intersects(G, enew):
    """Given a graph G and a potential new edge enew,
    check if enew will intersect any old edge.
    """
    E1 = MyPoint(enew[0], enew[1])
    E2 = MyPoint(enew[2], enew[3])
    for e in G.es():
        O1 = MyPoint(e.source_vertex["x"], e.source_vertex["y"])
        O2 = MyPoint(e.target_vertex["x"], e.target_vertex["y"])
        if segments_intersect(E1, E2, O1, O2):
            return True
    return False
    

def delete_overlaps(G_res, G_orig, verbose = False):
    """Deletes inplace all overlaps of G_res with G_orig (from G_res)
    based on node ids. In other words: G_res -= G_orig
    """
    del_edges = []
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            # If there is already an edge in the original network, delete it
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                del_edges.append(e.index)
        except:
            pass
    G_res.delete_edges(del_edges)
    # Remove isolated nodes
    #isolated_nodes = G_res.vs.select(_degree_eq=0)
    #G_res.delete_vertices(isolated_nodes)
    if verbose: print("Removed " + str(len(del_edges)) + " overlapping edges and " + str(len(isolated_nodes)) + " nodes.")

def constrict_overlaps(G_res, G_orig, factor = 5):
    """Increases length by factor of all overlaps of G_res with G_orig (in G_res) based on edge ids.
    """
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                G_res.es[e.index]["weight"] = factor * G_res.es[e.index]["weight"]
        except:
            pass



    
def greedy_triangulation(GT, GT_NOex, poipairs, D = 1000, existing_edge_length = 0,node_weights = None, existing_edges = None,poidist_and_alpha = [500,0.5]):
    """Greedy Triangulation (GT) of a graph GT with an empty edge set.
    Distances between pairs of nodes are given by poipairs.
    
    The GT connects pairs of nodes in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    """
    osmid_list = GT.es['osmid']
    osmid_list_nodes = GT.vs['id']
    
    
    # define poi_distance and alpha
    poi_distance = poidist_and_alpha[0] # delta param
    alpha = poidist_and_alpha[1] # alpha param
    
    # import the count of accidents and trips through links
    path_file = PATH["data"] + placeid + "/" + placeid + '_ONLYaccidentsrouting_counter' + str(poi_distance) + '.csv'
    df1 = pd.read_csv(path_file)
    path_file = PATH["data"] + placeid + "/" + placeid + '_ONLYtrips_counter' + str(poi_distance) + '.csv'
    df2 = pd.read_csv(path_file)
    
    # create a dict with: keys = "poipair[0],poipair[1]" and value = [n_accidents,n_trips]
    counter_dict = {}
    for i in range(len(df1)):
        counter_dict[df1.iloc[i]['poipairs']] = [df1.iloc[i]['accidents'],df2.iloc[i]['trips']]
    counter_dict['acc_max,trips_max'] = [max(df1['accidents']),max(df2['trips'])]
    
    # define the number of accidents and trips per km
    acc_perKM_list = []
    trips_perKM_list = []
    for poipair, poipair_distance in poipairs:
        acc_perKM = counter_dict[str(poipair[0])+','+str(poipair[1])][0]*1000/poipair_distance
        acc_perKM_list.append(acc_perKM)
        trips_perKM = counter_dict[str(poipair[0])+','+str(poipair[1])][1]*1000/poipair_distance
        trips_perKM_list.append(trips_perKM)
        
    # calculate the max of accidents and trips per km
    max_acc = max(acc_perKM_list)
    max_trips = max(trips_perKM_list)
    
    for poipair, poipair_distance in poipairs:
        poipair_ind = (GT.vs.find(id = poipair[0]).index, GT.vs.find(id = poipair[1]).index)
        poipair_ind_NOex = (GT_NOex.vs.find(id = poipair[0]).index, GT_NOex.vs.find(id = poipair[1]).index)
        if not new_edge_intersects(GT_NOex, (GT.vs[poipair_ind[0]]["x"], GT.vs[poipair_ind[0]]["y"], GT.vs[poipair_ind[1]]["x"], GT.vs[poipair_ind[1]]["y"])):
            
            # set the link weight
            if node_weights != None:
                
                # for each link define count the number of accidents/trips per 1000 meters (if we simply count the number of accidents/trips through each link, we advantage the longer links)
                acc_perKM = counter_dict[str(poipair[0])+','+str(poipair[1])][0]*1000/poipair_distance
                trips_perKM = counter_dict[str(poipair[0])+','+str(poipair[1])][1]*1000/poipair_distance
                
                
                w_acc = (poipair_distance + 1)/(1 + 9*acc_perKM/max_acc)
                w_trips = (poipair_distance + 1)/(1 + 9*trips_perKM/max_trips)
                
                # combine the contribution of accidents and trips data
                w = alpha*w_trips + (1-alpha)*w_acc
                
                GT.add_edge(poipair_ind[0], poipair_ind[1], weight = w,length = poipair_distance) 
                GT_NOex.add_edge(poipair_ind_NOex[0], poipair_ind_NOex[1], weight = w,length = poipair_distance)
            
            # if no data are provided the prioritization is based on the infrastructure
            else:
                GT.add_edge(poipair_ind[0], poipair_ind[1], weight = poipair_distance)
                GT_NOex.add_edge(poipair_ind_NOex[0], poipair_ind_NOex[1], weight = poipair_distance)

    # Set the km of investments
    L = edge_lengths(GT) - existing_edge_length - D
   
    # Get the measure
    BW = GT.edge_betweenness(directed = False, weights = "weight")
    argsorted_BW = np.argsort(BW)
        
    removed = []
    removed_NOex = []
    for i in range(len(argsorted_BW)):
            
        if i < 2:
            removed.append(int(argsorted_BW[i]))
            removed_NOex.append(GT_NOex.es.find('osmid' == GT.es[int(argsorted_BW[i])]['osmid']).index) 
            
        elif sum(GT.es[removed]['length']) < L and GT.es[int(argsorted_BW[i])]['osmid'] not in osmid_list:
            removed.append(int(argsorted_BW[i]))
            removed_NOex.append(GT_NOex.es.find('osmid' == GT.es[int(argsorted_BW[i])]['osmid']).index)
           
    sub_edges = []     
         
    for c,e in enumerate(GT.es):

        GT.es[c]['bw'] = BW[c]
        

    GT.es[removed].delete()
    GT_NOex.es[removed_NOex].delete()
        
    print('Length: ', edge_lengths(GT) - existing_edge_length)
     
    return GT,BW,GT_NOex


def greedy_triangulation_routing(G, pois, Ds = [1],node_weights = None, existing = False, G_bike = None, poidist_and_alpha = [500,0.5], limited_abstract_edges=None):
    """Greedy Triangulation (GT) of a graph G's node subset pois,
    then routing to connect the GT (up to D km of investments).
    G is an ipgraph graph, pois is a list of node ids.
    
    The GT connects pairs of nodes in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    
    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """
    poi_distance = poidist_and_alpha[0] # delta parameter
    alpha = poidist_and_alpha[1] # alpha parameter
    existing_edge_length = sum(G_bike.es['weight']) # length of existing bike net
    
    # import the trips and accidents counted on existing infrastructure
    path_file = PATH["data"] + placeid + "/" + placeid + '_ONLYaccidentsrouting_EXISTING_counter.csv'
    df1 = pd.read_csv(path_file)
    path_file = PATH["data"] + placeid + "/" + placeid + '_ONLYtrips_EXISTING_counter.csv'
    df2 = pd.read_csv(path_file)
    
    # create a dict with: keys = "poipair[0],poipair[1]" and value = [n_accidents,n_trips]
    existing_dict = {}
    for i in range(len(df1)):
        existing_dict[str(df1.iloc[i]['link'])] = [df1.iloc[i]['accidents'],df2.iloc[i]['trips']]
    existing_dict['acc_max,trips_max'] = [max(df1['accidents']),max(df2['trips'])]
    
    # define the number of accidents and trips per km
    acc_perKM_list = []
    trips_perKM_list = []
    for key in existing_dict.keys():
        if key != 'acc_max,trips_max':
            acc_perKM = existing_dict[key][0]*1000/G_bike.es[int(key)]['weight']
            acc_perKM_list.append(acc_perKM)
            trips_perKM = existing_dict[key][1]*1000/G_bike.es[int(key)]['weight']
            trips_perKM_list.append(trips_perKM)
    
    # calculate the max of accidents and trips per km
    max_acc_ex = max(acc_perKM_list)
    max_trips_ex = max(trips_perKM_list)    
    
    if len(pois) < 2: return ([], []) # We can't do anything with less than 2 POIs

    # GT_abstract is the GT with same nodes but euclidian links to keep track of edge crossings
    pois_indices = set()
    for poi in pois:
        pois_indices.add(G.vs.find(id = poi).index)
    
    existing_edges = []
    
    G_temp = copy.deepcopy(G)

    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
            
    # add the edges of the existing infrastructure to the initial solution
    for e in G.es:    
            
        # break if we already added all the edges
        if len(G_temp.es) >= len(G_bike.es):
            break
    
        # add to G_temp the edges that are in both G_bike and G_carall
        for k in range(len(G_bike.es)):
            if e.attributes() == G_bike.es[k].attributes():
                    
                acc_perKM = existing_dict[str(k)][0]*1000/G_bike.es[k]['weight']
                trips_perKM = existing_dict[str(k)][1]*1000/G_bike.es[k]['weight']
                
                w_acc = (G_bike.es[k]['weight'] + 1)/(1 + 9*acc_perKM/max_acc_ex)
                w_trips = (G_bike.es[k]['weight'] + 1)/(1 + 9*trips_perKM/max_trips_ex)
                
                # combine the contribution of accidents and trips data
                w = alpha*w_trips + (1-alpha)*w_acc
                        
                G_temp.add_edge(e.source,e.target,weight=w,length=e['weight'],osmid = e['osmid'])
                existing_edges.append((e.source,e.target,e['weight'],e['osmid']))
                break
                    
    # add to a set the nodes of the extremes in the existing bike infrastructure
    vv = set() # we will use this set to create a GT_abstract (euclidian links)
    for e in G_temp.es: #ho cambiato Gcar_temp in G_temp
        vv.add(e.source)
        vv.add(e.target)
        
    # add also the pois to this set
    for poi in pois:
        vv.add(G.vs.find(id = poi).index)
    
      
    poipairs = poipairs_by_distance(G, pois, True)
    
    # limit the seed points pairs to abstract links with length < limited_abstract_edges meters
    # we suggest to set limited_abstract_edges meters != None (e.g. 5000 meters) for big cities for time computation reasons
    if limited_abstract_edges != None:
        for ix,poipair in enumerate(poipairs):
            if poipair[1] > limited_abstract_edges:
                poipairs = copy.deepcopy(poipairs[:ix])
                break
    
    
    if len(poipairs) == 0: return ([], [])
    
    GT_abstracts = []
    GTs = []
    BWs = []
    for D in tqdm(Ds, desc = "Greedy triangulation", leave = False):
        
        #if existing:
        # the initial solution it is not empty
        GT_abstract = copy.deepcopy(G_temp.subgraph(vv))
            
        # we try to avoid the edge_crossing constrait on the existing net by passing another GT_abstract to greedy_triang() without the existing net
            
        G_temp_noexisting = copy.deepcopy(G)
        for e in G_temp_noexisting.es: # delete all edges
            G_temp_noexisting.es.delete(e)
                
        GT_abstract_noexisting = copy.deepcopy(G_temp_noexisting.subgraph(pois_indices))
        
        # delete the existing infrastracture, we use GT_abstract_noexisting for the edge crossing constrait
        # GT_abstract_noexisting = copy.deepcopy(GT_abstract)
        #    delete_overlaps(GT_abstract_noexisting, G_bike, verbose = False)
               
        GT_abstract,BW,GT_abstract_noexisting = greedy_triangulation(GT_abstract, GT_abstract_noexisting, poipairs, D, existing_edge_length, node_weights, existing_edges,poidist_and_alpha)
        GT_abstracts.append(GT_abstract)
        BWs.append(BW)
        
        
        # Get node pairs we need to route, sorted by distance
        routenodepairs = {}
        for e in GT_abstract.es:
            routenodepairs[(e.source_vertex["id"], e.target_vertex["id"])] = e["length"]
        routenodepairs = sorted(routenodepairs.items(), key = lambda x: x[1])

        # Do the routing
        GT_indices = set()
        for poipair, poipair_distance in routenodepairs:
            poipair_ind = (G.vs.find(id = poipair[0]).index, G.vs.find(id = poipair[1]).index)
            sp = set(G.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
            GT_indices = GT_indices.union(sp)

        GT = G.induced_subgraph(GT_indices)
        GTs.append(GT)
    
    return (GTs, GT_abstracts,BWs)
    
    
def poipairs_by_distance(G, pois, return_distances = False):
    """Calculates the (weighted) graph distances on G for a subset of nodes pois.
    Returns all pairs of poi ids in ascending order of their distance. 
    If return_distances, then distances are also returned.
    """
    
    # Get poi indices
    #print('Get poi indices...')
    indices = []
    for poi in pois:
        indices.append(G.vs.find(id = poi).index)
    
    # Get sequences of nodes and edges in shortest paths between all pairs of pois
    
    poi_nodes = []
    poi_edges = []
    for c, v in enumerate(indices):

        poi_nodes.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "vpath"))
        poi_edges.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "epath"))

    
    # Sum up weights (distances) of all paths
    poi_dist = {}
    k = 1
    for paths_n, paths_e in zip(poi_nodes, poi_edges):

        for path_n, path_e in zip(paths_n, paths_e):
            # Sum up distances of path segments from first to last node
            path_dist = sum([G.es[e]['weight'] for e in path_e])
            if path_dist > 0:
                poi_dist[(path_n[0],path_n[-1])] = path_dist
        
        k = k + 1
     
            
    temp = sorted(poi_dist.items(), key = lambda x: x[1])
    # Back to ids
    output = []
    for ii,p in enumerate(temp):
     
        output.append([(G.vs[p[0][0]]["id"], G.vs[p[0][1]]["id"]), p[1]])
      
    if return_distances:
        return output
    else:
        return [o[0] for o in output]


def rotate_grid(p, origin = (0, 0), degrees = 0):
        """Rotate a list of points around an origin (in 2D). 
        
        Parameters:
            p (tuple or list of tuples): (x,y) coordinates of points to rotate
            origin (tuple): (x,y) coordinates of rotation origin
            degrees (int or float): degree (clockwise)

        Returns:
            ndarray: the rotated points, as an ndarray of 1x2 ndarrays
        """
        # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
        angle = np.deg2rad(-degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)


# Two functions from: https://github.com/gboeing/osmnx-examples/blob/v0.11/notebooks/17-street-network-orientations.ipynb
def reverse_bearing(x):
    return x + 180 if x < 180 else x - 180

def count_and_merge(n, bearings):
    # make twice as many bins as desired, then merge them in pairs
    # prevents bin-edge effects around common values like 0° and 90°
    n = n * 2
    bins = np.arange(n + 1) * 360 / n
    count, _ = np.histogram(bearings, bins=bins)
    
    # move the last bin to the front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    return count[::2] + count[1::2]


def calculate_directness(G, numnodepairs = 500):
    """Calculate directness on G over all connected node pairs in indices.
    """
    
    indices = random.sample(list(G.vs), min(numnodepairs, len(G.vs)))

    poi_edges = []
    v1 = []
    v2 = []
    total_distance_direct = 0
    for c, v in enumerate(indices):
        poi_edges.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "epath"))
        temp = G.get_shortest_paths(v, indices[c:], weights = "weight", output = "vpath")
        total_distance_direct += sum(dist_vector([(G.vs[t[0]]["y"], G.vs[t[0]]["x"]) for t in temp], [(G.vs[t[-1]]["y"], G.vs[t[-1]]["x"]) for t in temp])) # must be in format lat,lon = y, x
    
    total_distance_network = 0
    for paths_e in poi_edges:
        for path_e in paths_e:
            # Sum up distances of path segments from first to last node
            total_distance_network += sum([G.es[e]['weight'] for e in path_e])
    
    return total_distance_direct / total_distance_network


def listmean(lst): 
    try: return sum(lst) / len(lst)
    except: return 0

def calculate_coverage_edges(G, buffer_m = 500, return_cov = False, G_prev = ig.Graph(), cov_prev = Polygon()):
    """Calculates the area and shape covered by the graph's edges.
    If G_prev and cov_prev are given, only the difference between G and G_prev are calculated, then added to cov_prev.
    """

    G_added = copy.deepcopy(G)
    delete_overlaps(G_added, G_prev)

    # https://gis.stackexchange.com/questions/121256/creating-a-circle-with-radius-in-metres
    loncenter = listmean([v["x"] for v in G.vs])
    latcenter = listmean([v["y"] for v in G.vs])
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    aeqd_to_wgs84 = pyproj.Transformer.from_proj(
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"))
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G_added.es]
    # Shapely buffer seems slow for complex objects: https://stackoverflow.com/questions/57753813/speed-up-shapely-buffer
    # Therefore we buffer piecewise.
    cov_added = Polygon()
    for c, t in enumerate(edgetuples):
        # if cov.geom_type == 'MultiPolygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), sum([len(pol.exterior.coords) for pol in cov]))
        # elif cov.geom_type == 'Polygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), len(pol.exterior.coords))
        buf = ops.transform(aeqd_to_wgs84.transform, ops.transform(wgs84_to_aeqd.transform, LineString(t)).buffer(buffer_m))
        cov_added = ops.unary_union([cov_added, Polygon(buf)])

    # Merge with cov_prev
    if not cov_added.is_empty: # We need this check because apparently an empty Polygon adds an area.
        cov = ops.unary_union([cov_added, cov_prev])
    else:
        cov = cov_prev

    cov_transformed = ops.transform(wgs84_to_aeqd.transform, cov)
    covered_area = cov_transformed.area / 1000000 # turn from m2 to km2

    if return_cov:
        return (covered_area, cov)
    else:
        return covered_area


def calculate_poiscovered(G, cov, nnids):
    """Calculates how many nodes, given by nnids, are covered by the shapely (multi)polygon cov
    """
    
    pois_indices = set()
    for poi in nnids:
        pois_indices.add(G.vs.find(id = poi).index)

    poiscovered = 0
    for poi in pois_indices:
        v = G.vs[poi]
        if Point(v["x"], v["y"]).within(cov):
            poiscovered += 1
    
    return poiscovered

def calculate_metrics(G, GT_abstract, G_big, nnids, OD=None,calcmetrics = {
          "coverage": 0,
          "directness_lcc": 0,
          "trips_coverage": 0,
          "accidents_coverage19" : 0,
          "components": 0
         }, buffer_walk = 500, buffer_accidents = 50, numnodepairs = 500, verbose = False, G_prev = ig.Graph(), cov_prev = Polygon(), cov_prev_acc = Polygon(), ignore_GT_abstract = False,G_big_OX=None,detour = 0):
    """Calculates all metrics (using the keys from calcmetrics).
    """
    # Seed for trips coverage
    ss = 10
    
    # Load accidents 2020
    accidents20 = pd.read_csv(PATH["data"] + placeid + "/" +'accidents_softmobility2020.csv')
    accidents20 = accidents20.drop(['Unnamed: 0'],axis=1)
        
    # Load accidents 2019
    accidents19 = pd.read_csv(PATH["data"] + placeid + "/" +'accidents_softmobility2019.csv')
    accidents19 = accidents19.drop(['Unnamed: 0'],axis=1)
    
    # Initialize cov and cov_acc
    cov = cov_prev
    cov_acc = cov_prev_acc
    return_cov = True
    
    output = {}
    for key in calcmetrics:
        output[key] = 0
    

    # Check that the graph has links (sometimes we have an isolated node)
    if G.ecount() > 0 and GT_abstract.ecount() > 0:

        # Get LCC
        cl = G.clusters()
        LCC = cl.giant()

        # COVERAGE
        if "coverage" in calcmetrics:
            if verbose: print("Calculating coverage...")
            # G_added = G.difference(G_prev) # This doesnt work
            covered_area, cov = calculate_coverage_edges(G, buffer_walk, return_cov, G_prev, cov_prev)
            output["coverage"] = covered_area
            
        # POI COVERAGE
        if "poi_coverage" in calcmetrics:
            if verbose: print("Calculating POI coverage...")
            output["poi_coverage"] = calculate_poiscovered(G_big, cov, nnids)
            
        # ACCIDENTS COVERAGE
        if "accidents_coverage19" in calcmetrics or "accidents_coverage" in calcmetrics:
            
            covered_area, cov_acc = calculate_coverage_edges(G, buffer_accidents, return_cov, G_prev, cov_prev_acc)
            
            # ACCIDENTS COVERAGE 2020
            if "accidents_coverage20" in calcmetrics:
                if verbose: print("Calculating crash coverage 2020...")
       
                output["accidents_coverage20"] = calculate_accidentscovered(cov_acc, accidents20)
            
            # ACCIDENTS COVERAGE 2019
            if "accidents_coverage19" in calcmetrics or "accidents_coverage" in calcmetrics:
                if verbose: print("Calculating crash coverage...")
            
                output["accidents_coverage19"] = calculate_accidentscovered(cov_acc, accidents19)        
        
        # TRIPS COVERAGE
        if "trips_coverage" in calcmetrics:
            if verbose: print("Calculating trips coverage...")
            output["trips_coverage"] = calculate_tripscovered(G_big,G_big_OX, G, numnodepairs, OD, ss,detour)
        
        # COMPONENTS
        if "components" in calcmetrics:
            if verbose: print("Calculating components...")
            output["components"] = len(list(G.components()))
        
        # DIRECTNESS
        if verbose and ("directness" in calcmetrics or "directness_lcc" in calcmetrics): print("Calculating directness...")
        if "directness" in calcmetrics:
            output["directness"] = calculate_directness(G, numnodepairs)
        if "directness_lcc" in calcmetrics:
            if len(cl) > 1:
                output["directness_lcc"] = calculate_directness(LCC, numnodepairs)
            else:
                output["directness_lcc"] = output["directness"]

    return (output, cov, cov_acc)
    

def overlap_linepoly(l, p):
    """Calculates the length of shapely LineString l falling inside the shapely Polygon p
    """
    return p.intersection(l).length if l.length else 0


def edge_lengths(G,weights=True):
    """Returns the total length of edges in an igraph graph.
    """
    if weights == False: return sum([e['weight'] for e in G.es])
    else: return sum([e['length'] for e in G.es])

def intersect_igraphs(G1, G2):
    """Generates the graph intersection of igraph graphs G1 and G2, copying also link and node attributes.
    """
    # Ginter = G1.__and__(G2) # This does not work with attributes.
    if G1.ecount() > G2.ecount(): # Iterate through edges of the smaller graph
        G1, G2 = G2, G1
    inter_nodes = set()
    inter_edges = []
    inter_edge_attributes = {}
    inter_node_attributes = {}
    edge_attribute_name_list = G2.edge_attributes()
    node_attribute_name_list = G2.vertex_attributes()
    for edge_attribute_name in edge_attribute_name_list:
        inter_edge_attributes[edge_attribute_name] = []
    for node_attribute_name in node_attribute_name_list:
        inter_node_attributes[node_attribute_name] = []
    for e in list(G1.es):
        n1_id = e.source_vertex["id"]
        n2_id = e.target_vertex["id"]
        try:
            n1_index = G2.vs.find(id = n1_id).index
            n2_index = G2.vs.find(id = n2_id).index
        except ValueError:
            continue
        if G2.are_connected(n1_index, n2_index):
            inter_edges.append((n1_index, n2_index))
            inter_nodes.add(n1_index)
            inter_nodes.add(n2_index)
            edge_attributes = e.attributes()
            for edge_attribute_name in edge_attribute_name_list:
                inter_edge_attributes[edge_attribute_name].append(edge_attributes[edge_attribute_name])

    # map nodeids to first len(inter_nodes) integers
    idmap = {n_index:i for n_index,i in zip(inter_nodes, range(len(inter_nodes)))}

    G_inter = ig.Graph()
    G_inter.add_vertices(len(inter_nodes))
    G_inter.add_edges([(idmap[e[0]], idmap[e[1]]) for e in inter_edges])
    for edge_attribute_name in edge_attribute_name_list:
        G_inter.es[edge_attribute_name] = inter_edge_attributes[edge_attribute_name]

    for n_index in idmap.keys():
        v = G2.vs[n_index]
        node_attributes = v.attributes()
        for node_attribute_name in node_attribute_name_list:
            inter_node_attributes[node_attribute_name].append(node_attributes[node_attribute_name])
    for node_attribute_name in node_attribute_name_list:
        G_inter.vs[node_attribute_name] = inter_node_attributes[node_attribute_name]

    return G_inter


def calculate_metrics_additively(Gs, GT_abstracts, Ds, G_big, nnids, buffer_walk = 500, buffer_accidents = 50, numnodepairs = 500, verbose = False, output = {
            "coverage": [],
            "directness_lcc": [],
            "accidents_coverage19" : [],
            "trips_coverage" : [],
            "components": [],
              }, detour = 0, cov_prev = Polygon(), cov_prev_acc = Polygon()):
    """Calculates all metrics, additively. 
    Coverage differences are calculated in every step instead of the whole coverage.
    """
    # load OD data for the trip_coverage metric
    OD = pd.read_csv(PATH["data"] + placeid + "/" + "OD_data.csv")
    OD = OD.drop(['Unnamed: 0'], axis = 1)
    
    G_big_OX = csv_to_ox(PATH["data"] + placeid + "/", placeid, 'biketrackcarall')
    
    # BICYCLE NETWORKS
    covs = {} # covers using buffer_walk
    covs_acc = {} # covers using buffer_accident
    #cov_prev = Polygon()
    #cov_prev_acc = Polygon()
    GT_prev = ig.Graph()
    for GT, GT_abstract, D in zip(Gs, GT_abstracts, Ds):
        if verbose: print("Calculating bike network metrics for D = " + str(D), " m.")
        metrics, cov, cov_acc = calculate_metrics(GT, GT_abstract, G_big, nnids, OD, output, buffer_walk, buffer_accidents, numnodepairs, verbose, GT_prev, cov_prev, cov_prev_acc, False,G_big_OX,detour)
        print(metrics)
        for key in output.keys():
            output[key].append(metrics[key])
        covs[D] = cov
        covs_acc[D] = cov_acc
        cov_prev = copy.deepcopy(cov)
        cov_prev_acc = copy.deepcopy(cov_acc)
        GT_prev = copy.deepcopy(GT)
    return (output, covs, covs_acc)


def write_result(res, mode, placeid, suffix, dictnested = {}):
    """Write results (pickle or dict to csv)
    """
    if mode == "pickle":
        openmode = "wb"
    else:
        openmode = "w"

    #if poi_source:
    filename = placeid + "_" + suffix
    #else:
    #    filename = placeid + "_" + prune_measure + suffix

    with open(PATH["results"] + placeid + "/" + filename, openmode) as f:
        if mode == "pickle":
            pickle.dump(res, f)
        elif mode == "dict":
            w = csv.writer(f)
            w.writerow(res.keys())
            try: # dict with list values
                w.writerows(zip(*res.values()))
            except: # dict with single values
                w.writerow(res.values())
        elif mode == "dictnested":
            # https://stackoverflow.com/questions/29400631/python-writing-nested-dictionary-to-csv
            fields = ['network'] + list(dictnested.keys())
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for key, val in sorted(res.items()):
                row = {'network': key}
                row.update(val)
                w.writerow(row)


def gdf_to_geojson(gdf, properties):
    """Turn a gdf file into a GeoJSON.
    The gdf must consist only of geometries of type Point.
    Adapted from: https://geoffboeing.com/2015/10/exporting-python-data-geojson/
    """
    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in gdf.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point',
                               'coordinates':[]}}
        feature['geometry']['coordinates'] = [row.geometry.x, row.geometry.y]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson



def ig_to_shapely(G):
    """Turn an igraph graph G to a shapely LineString
    """
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G.es]
    G_shapely = LineString()
    for t in edgetuples:
        G_shapely = ops.unary_union([G_shapely, LineString(t)])
    return G_shapely


def calculate_accidentscovered(cov, accidents):
    """Calculates how many crashes are covered by the shapely (multi)polygon cov
    """
    accidentscovered = 0
    for i in range(len(accidents)):
        
        if Point(accidents.iloc[i]['longitude'], -accidents.iloc[i]['latitude']).within(cov):
            accidentscovered += 1
    
    return accidentscovered


def calculate_overlapGvsG(G1,G2,return_G,return_normalized):
    """Calculates the overlap between G1 and G2
    """
    
    G_intersect = intersect_igraphs(G1, G2)
    km_intersection = edge_lengths(G_intersect)
    
    if return_normalized:
        km_G1 = km_intersection/(edge_lengths(G1)+1)
        km_G2 = km_intersection/(edge_lengths(G2)+1)
        
        return km_intersection, km_G1, km_G2, G_intersect
        
    elif return_G: return km_intersection, G_intersect
    
    else: return km_intersection
    
    
def calculate_networkdifference(G1,G2):
    
    G1_copy = copy.deepcopy(G1)
    removed = []
    
    # add the edges of the existing infrastructure to the initial solution
    for i,e in enumerate(G1.es):    
       
     # add to G_temp the edges that are in both G_bike and G_carall
        for k in range(len(G2.es)):
            if e.attributes() == G2.es[k].attributes():
                removed.append(i)
                break
    
    G1_copy.es[removed].delete()
    
    print(len(G1.es) == len(G1_copy.es) + len(G2.es))
    print(len(G1.es),len(G1_copy.es),len(G2.es))
                
    return G1_copy


def calculate_tripscovered(G_BkCl,G_c,G1, howmany, OD, s, detour = 0):
    
    # G_BkCl = biketrackcarall network (IG format)
    # G_c = biketrackcarall network (OX format)
    # G1 = biketrack network
    # howmany = number of OD samples used to calculate trips coverage (we suggest howmany < 1000)
    # OD = origin destination data
    # s = random seed to select the OD data used for the calculation
    # detour = level of detour accepted, must be in [0,1]
    
    ### Implement the detour effect:
    # length of links not belonging to bike net are multiplied by (1+detour)
    G_btc_temp = copy.deepcopy(G_BkCl)
    
    if detour > 0:
    
    
        delete_overlaps(G_btc_temp, G1, verbose = False)

        G_btc_temp.es['weight'] = [n*(1+detour) for n in G_btc_temp.es['weight']]

        for i,e in enumerate(G1.es):
            G_btc_temp.add_edge(G_btc_temp.vs.find(id = G1.vs[e.source]['id']).index,G_btc_temp.vs.find(id = G1.vs[e.target]['id']).index,weight=e['weight'],osmid = e['osmid'])
    
    
    ### Select randomly howmany OD samples
    # it is important to use the same random seed used in "3c_crash&trips-per-link.ipynb" to calculate the number of trips per link. In this way here we can use "howmany" samples that are not used elsewhere.
    ids = list(np.arange(len(OD)))
    random.seed(s)
    random.shuffle(ids)
    selected_ids = ids[-howmany:] # make sure that you excluded at least "howmany" trips in "3c_crash&trips-per-link.ipynb", so the samples used here (to evaluate trip coverage) are not used to build the weighted distance in the model
    
    routenodepairs = list()

    # select the nearest node for each O,D coordinate
    for j,i in enumerate(selected_ids):
    
        nO=ox.distance.get_nearest_node(G_c,[OD['O_lat'].iloc[i],
                                                  OD['O_lon'].iloc[i]])
        nD=ox.distance.get_nearest_node(G_c, [OD['D_lat'].iloc[i],
                                                   OD['D_lon'].iloc[i]])
        routenodepairs.append((nO,nD))

        
    ## Snap the shortest paths on the road network
    
    tot_e_km = []
    bike_e_km = []
        

    for i,poipair in enumerate(routenodepairs):
        GT_indices = set()
        poipair_ind = (G_btc_temp.vs.find(id = poipair[0]).index, G_btc_temp.vs.find(id = poipair[1]).index)
        sp = set(G_btc_temp.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
        GT_indices = GT_indices.union(sp)
    
        GT = G_btc_temp.induced_subgraph(GT_indices)    
        GResult = intersect_igraphs(GT, G1)
    
    ## Append to the lists the length in km of trip i 
        if len(GT.es) > 0:
           
            tot_e_km.append(edge_lengths(GT,False)) # append the total length
            
            bike_e_km.append(edge_lengths(GResult,False)) # append the km traveled on biketracks
           
        else:
           
            tot_e_km.append(0)
            
            bike_e_km.append(0)            
    
    return sum(bike_e_km)/sum(tot_e_km)


def line_intersection(line1, line2):
    
    # This function find the intersection between 2 lines
    # A line is defined by the tuples [(x0,y0),(x1,y1)]
    # This function is used in function find_intersect() to find alpha tradeoff
    
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def find_intersect(T,C,alphas,C0=120,Ctot=314,T0=0.06411324702675697):
    
    # This function find alpha tradeoff
    # Used in notebook "plots.ipynb"
    
    # T = trips_coverage list
    # C = crashes_coverage list
    # alphas = alpha values list
    # C0 = crashes_coverage for D = 0 km
    # Ctot = total number of crashes
    # T0 = trips_coverage for D = 0 km
    
    c = [x/Ctot-C0/Ctot for x in C] 
    t = [x-T0 for x in T]
    
    for i in range(len(T)):
        if c[i]-t[i] > 0 and c[i+1]-t[i+1] < 0:
            line_c = [(alphas[i],c[i]),(alphas[i+1],c[i+1])]
            line_t = [(alphas[i],t[i]),(alphas[i+1],t[i+1])]
            break
    
    return line_intersection(line_c, line_t)


# this function creates a grid
def create_grid(xmin,xmax,ymin,ymax,n_cells):
    
    cell_size = (xmax-xmin)/n_cells
    
    # projection of the grid

    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

    # create the cells in a loop

    grid_cells = []
    
    for x0 in np.arange(xmin, xmax+cell_size, cell_size):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
        
        # bounds
        
            x1 = x0 - cell_size
            y1 = y0 + cell_size
        
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
            
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    return cell

# this function compute the distance in meters between 2 points 
def distance_meters(p_i, p_i_plus_1):
    points_df = gpd.GeoDataFrame({'geometry': [p_i, p_i_plus_1]}, crs='EPSG:4326')
    points_df = points_df.to_crs(5234)
    points_df2 = points_df.shift() #We shift the dataframe by 1 to align pnt1 with pnt2
    dist = points_df.distance(points_df2)
    
    return dist

## to collect e-scooter position data
# location request
def make_a_request(lat,lon,rad,j):
    
    # set headers
    headers = {
        'Authorization': 'Bearer '+j["access"],
        'User-Agent': 'Bird/4.119.0(co.bird.Ride; build:3; iOS 14.3.0) Alamofire/5.2.2',
        'legacyrequest': 'false',
        'Device-Id': 'BC5BDA72-12A2-5DF4-BD11-92ACF8FFC86D',
        'App-Version': '4.119.0',        
        'Location': json.dumps({"latitude": lat, "longitude": lon,"altitude":500,"accuracy":100,"speed":-1,"heading":-1}, sort_keys=True)
    }
    # set url
    URL = 'https://api-bird.prod.birdapp.com/bird/nearby?latitude=' + str(lat) + '&longitude=' + str(lon) + '&radius=' + str(rad)
    
    # make a GET request!
    r = requests.get(url=URL, headers=headers)
    
    return r

## to collect e-scooter position data
# refresh token
def refresh_token(j):
    POST = " https://api-auth.prod.birdapp.com/api/v1/auth/refresh/token"
    
    headers = {
    'User-Agent': 'Bird/4.119.0(co.bird.Ride; build:3; iOS 14.3.0) Alamofire/5.2.2',
    'Device-Id': '4984dbee-5300-40d6-bbdf-39c94ae4a39e',
    'Platform': 'ios',
    'App-Version': '4.119.0',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + j["refresh"]
    }
    r = requests.post(url=POST, headers=headers)
    print("token refreshed, status code: ", r.status_code)
    return r

## to create OD data 
# find unique values for 'code' in all df
# then create a list with all the 'code' unique values found in data collection

def find_unique_code(frames):
    code_list = []
    
    for i in range(len(frames)):
        a = frames[i]['code'].value_counts()
       
        for j in range(len(a)):    
            if a[j] < 2:
                code_list.append(a.keys()[j])
                
    return pd.DataFrame(code_list)[0].unique()

## to create OD data 
# remove the data in which 'code' refears to more than 1 e-scooter across all frames
def remove_doubles(frames,unique_code):

    code_list = []
    doubles = []

    for i in range(len(frames)):
        print('List of doubles - Analyzed frames: ',i+1,'/',len(frames))
        a = frames[i]['code'].value_counts()
        
        for j in range(len(a)):
            if a[j] > 1 and a.keys()[j] not in doubles:
                doubles.append(a.keys()[j])

        clear_output(wait = True)
    
    for i in range(len(unique_code)):
        print('List non-double codes - Analyzed codes: ',i+1,'/',len(unique_code))
        if not unique_code[i] in doubles:
            code_list.append(unique_code[i])
        clear_output(wait = True)
        
    return code_list, doubles


# This function finds all the movements of e-scooters and creates an OD-dataset
# ARGUMENTS: frames = list of TimeSlots (list of df), code = code of e-scooter (string), t_space = threshold for the movement (in meters), t_time = threshold for time in minutes (int) 
def OD_data(frames,code,t_space,t_time):
    
    seq_TF01_code = find_ID_path(code, frames)
   
    O_lat = []
    O_lon = []
    O_time = []
    O_battery = []
    D_lat = []
    D_lon = []
    D_time = []
    D_battery = []
    codes = []
    
    for i in range(1,len(frames)):
        # check that 'code' is present in both slots i and i-1
        if seq_TF01_code[i] != -1 and seq_TF01_code[i-1] != -1:
            
            # compute the distance between the position in the slot i and i-1
            delta_space = haversine((frames[i].iloc[seq_TF01_code[i]]['latitude'], frames[i].iloc[seq_TF01_code[i]]['longitude']), (frames[i-1].iloc[seq_TF01_code[i-1]]['latitude'], frames[i-1].iloc[seq_TF01_code[i-1]]['longitude']), unit="m")
            
            # add to the OD data if there is a movement (1st condition) and if the time between the slots is < t_time (2nd condition)
            if delta_space > t_space and (frames[i]['Time'].iloc[0]-frames[i-1]['Time'].iloc[0])/60 < t_time:
                O_lat.append(frames[i-1].iloc[seq_TF01_code[i-1]]['latitude'])
                O_lon.append(frames[i-1].iloc[seq_TF01_code[i-1]]['longitude'])
                O_time.append(frames[i-1].iloc[seq_TF01_code[i-1]]['Time'])
                O_battery.append(frames[i-1].iloc[seq_TF01_code[i-1]]['battery_level'])
                D_lat.append(frames[i].iloc[seq_TF01_code[i]]['latitude'])
                D_lon.append(frames[i].iloc[seq_TF01_code[i]]['longitude'])
                D_time.append(frames[i].iloc[seq_TF01_code[i]]['Time'])
                D_battery.append(frames[i].iloc[seq_TF01_code[i]]['battery_level'])
                codes.append(code)
 
    return O_lat,O_lon,O_time,O_battery,D_lat,D_lon,D_time,D_battery,codes

# this function finds the index of an ID in each Slot, if ID is not in a slot, it return index -1
# ID: is the id that we want to find in the slots (is a string)
# frames: is a list of df, each df is a slot
# returns  a list of indices
def find_ID_path(ID, frames):
    
    id_seq = []
        
    for j in range(len(frames)):
            
    # find the index of the ID in all frames and append it to a list
        a = getIndexes(frames[j], ID)
        if len(a) > 0:
            id_seq.append(a[0][0])
    # if we find that in 1 frame the ID is not present, break
        else:
            id_seq.append(-1)
    
    return id_seq

# This function will return a list of
# positions where element exists
# in the dataframe.
## https://thispointer.com/python-find-indexes-of-an-element-in-pandas-dataframe/
def getIndexes(dfObj, value):
      
    # Empty list
    listOfPos = []
      
    # isin() method will return a dataframe with 
    # boolean values, True at the positions    
    # where element exists
    result = dfObj.isin([value])
      
    # any() method will return 
    # a boolean series
    seriesObj = result.any()
  
    # Get list of column names where 
    # element exists
    columnNames = list(seriesObj[seriesObj == True].index)
     
    # Iterate over the list of columns and
    # extract the row index where element exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
  
        for row in rows:
            listOfPos.append((row, col))
              
    # This list contains a list tuples with 
    # the index of element in the dataframe
    return listOfPos


print("Loaded functions.\n")
