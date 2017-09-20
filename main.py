import sys
import queue
import copy
import time

from pyclustering.cluster.optics import optics
from pyclustering.cluster import cluster_visualizer
import kdtree
from rtree import index
from rtree.index import Rtree

import numpy as np
from mpi4py import MPI

from scipy.spatial.distance import euclidean

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

dimension = 2

eps = None
minpts = None

def extended_mbr(points, eps):
    bounds = []
    for i in range(dimension):
        coords = [coord[i] for coord in points]
        bounds.append((min(coords)-eps, max(coords)+eps))
    return bounds

def get_neighbors(point, points, rtree):
    bounds = []
    for i in range(dimension):
        bounds.append([point[i] - eps, point[i] + eps])
    new_bounds = []
    for i in range(2):
        for j in range(dimension):
            new_bounds.append(bounds[j][i])
    neighbors = rtree.intersection(new_bounds)
    results = []
    for neighbor in neighbors:
        if len([obj for obj in points if obj.index_object == neighbor]) > 0:
            neighbor = [obj for obj in points if obj.index_object == neighbor][0]
            dist = euclidean(point, neighbor.point)
            if dist < eps:
                results.append(neighbor)
    return results

def minpts_distance(obj, neighbors):
    neighbors.sort(key=lambda x:euclidean(obj.point, x.point))
    minpts_distance = euclidean(obj.point, neighbors[minpts-1].point)
    return minpts_distance

def mark_affected_points(dict_objects1, dict_objects2, rtree1, rtree2):
    points1 = [x.point for x in dict_objects1.values()]
    points2 = [x.point for x in dict_objects2.values()]
    bounds1 = extended_mbr(points1, eps)
    bounds2 = extended_mbr(points2, eps)
    new_bounds1 = []
    new_bounds2 = []
    for i in range(2):
        for j in range(dimension):
            new_bounds1.append(bounds1[j][i])
            new_bounds2.append(bounds2[j][i])
    candidates1 = list(rtree1.intersection(new_bounds2))
    candidates2 = list(rtree2.intersection(new_bounds1))
    for x in candidates1:
        x_pt = dict_objects1[x].point
        for y in candidates2:
            y_pt = dict_objects2[y].point
            dist = euclidean(x_pt, y_pt)
            if dist < eps:
                dict_objects1[x].affected = True
                dict_objects1[x].affected = True
                """for obj in objects1:
                    if obj.index_object == x:
                        obj.affected = True
                for obj in objects2:
                    if obj.index_object == y:
                        obj.affected = True"""
    return

def update(obj, neighbors, pri_queue):
    return

def process_affected_point(dict_objects1, dict_objects2, rtree1, rtree2, obj, pri_queue, dest_objects):
    #print("PROCESS AFFECTED POINT")
    objects1 = dict_objects1.values()
    objects2 = dict_objects2.values()
    neighbors1 = get_neighbors(obj.point, objects1, rtree1)
    neighbors2 = get_neighbors(obj.point, objects2, rtree2)
    neighbors1.extend(neighbors2)
    neighbors = neighbors1
    if len(neighbors) >= minpts:
        obj.core_distance = minpts_distance(obj, neighbors)
        update(obj, neighbors, pri_queue)
        dest_objects.append(obj)
    obj.processed = True
    dict_objects1[obj.index_object].processed = True
    """for obj2 in objects1:
        if obj.index_object == obj2.index_object:
            obj2.processed = True"""
    

def predecessor(obj, objects1, objects2, rtree1, rtree2):
    neighbors1 = get_neighbors(obj.point, objects1, rtree1)
    neighbors2 = get_neighbors(obj.point, objects2, rtree2)
    neighbors1.extend(neighbors2)
    neighbors = neighbors1
    for neighbor in neighbors:
        if obj.core_distance is not None:
            if obj.reachability_distance == max([obj.core_distance, euclidean(obj.point, neighbor.point)]):
                return neighbor
        else:
            if obj.reachability_distance == euclidean(obj.point, neighbor.point):
                return neighbor
    #[neighbor for neighbor in neighbors if obj.reachability_distance == max([obj.core_distance, euclidean(obj.point, neighbor.point)])]

def successors(obj, objects1, objects2, rtree1, rtree2, neighbors):
    ss = []
    for neighbor in neighbors:
        pre = predecessor(neighbor, objects1, objects2, rtree1, rtree2)
        if pre is not None:
            if obj.index_object == pre.index_object:
                ss.append(neighbor)
    return ss

def exists_in_queue(obj, q):
    q2 = queue.PriorityQueue()
    q2.queue = copy.deepcopy(q.queue)
    while q2.qsize() != 0:
        top = q2.get()
        if top.index_object == obj.index_object:
            return True
    return False

def update_pq(target, pri_queue):
    q2 = queue.PriorityQueue()
    while pri_queue.qsize():
        top = pri_queue.get()
        if top.index_object == target.index_object:
            top.reachability_distance = target.reachability_distance
        q2.put(top)
    return q2

def reachdist(obj, origin, objects1, objects2, rtree1, rtree2):
    neighbors1 = get_neighbors(origin.point, objects1, rtree1)
    neighbors2 = get_neighbors(origin.point, objects2, rtree2)
    neighbors1.extend(neighbors2)
    neighbors = neighbors1
    reachdist = None
    if len(neighbors) >= minpts:
        reachdist = max(minpts_distance(obj, neighbors), euclidean(obj.point, origin.point))
    return reachdist

def process_nonaffected_point(dict_objects1, dict_objects2, rtree1, rtree2, obj, pri_queue, dest_objects):
    #print("PROCESS NON AFFECTED POINT")
    objects1 = dict_objects1.values()
    objects2 = dict_objects2.values()
    neighbors1 = get_neighbors(obj.point, objects1, rtree1)
    neighbors2 = get_neighbors(obj.point, objects2, rtree2)
    neighbors1.extend(neighbors2)
    neighbors = neighbors1
    ss = successors(obj, objects1, objects2, rtree1, rtree2, neighbors)
    ps = predecessor(obj, objects1, objects2, rtree1, rtree2)
    if ps is not None:
        ss.append(ps)
    targets = ss
    for target in targets:
        rdist = reachdist(target, obj, objects1, objects2, rtree1, rtree2)
        if target.processed == True:
            continue
        if not exists_in_queue(target, pri_queue):
            target.reachability_distance = rdist
            pri_queue.put(target)
        elif rdist is None:
            continue
        elif rdist < target.reachability_distance:
            target.reachability_distance = rdist
            pri_queue = update_pq(target, pri_queue)
    dest_objects.append(obj)
    obj.processed = True
    dict_objects1[obj.index_object].processed = True
    """for obj2 in objects1:
        if obj.index_object == obj2.index_object:
            obj2.processed = True"""
    return

def process(dict_objects1, dict_objects2, rtree1, rtree2, obj, pri_queue, dest_objects):
    if obj.affected:
        process_affected_point(dict_objects1, dict_objects2, rtree1, rtree2, obj, pri_queue, dest_objects)
    else:
        process_nonaffected_point(dict_objects1, dict_objects2, rtree1, rtree2, obj, pri_queue, dest_objects)
    return

def process_co(dict_objects1, dict_objects2, rtree1, rtree2, dest_objects):
    pq = queue.PriorityQueue()
    #while len([obj for obj in objects1 if obj.processed == False]) > 0:
    while len([obj for obj in dict_objects1.values() if obj.processed == False and obj.affected == True]) > 0:
        if pq.qsize() > 0:
            q = pq.get()
            process(dict_objects1, dict_objects2, rtree1, rtree2, q, pq, dest_objects)
        else:
            for obj in [obj for obj in dict_objects1.values() if obj.processed == False ]:
                if obj.processed:
                    continue
            #while len([obj for obj in co1 if obj.processed == False ]) > 0:
                #if rank == 0:
                #    print("len = %s" % len([obj for obj in co1 if obj.processed == False ]))
                #obj = [obj for obj in co1 if obj.processed == False ][0]
                #if rank == 0:
                #    print([obj for obj in co1 if obj.affected == True ])
                if obj.affected:
                    process_affected_point(dict_objects1, dict_objects2, rtree1, rtree2, obj, pq, dest_objects)
                    if pq.qsize() != 0:
                        break
    """
    for obj in [obj for obj in objects1 if obj.processed == False ]:
        if rank == 0:
            print("here rest = %s" % len([obj for obj in objects1 if obj.processed == False ]))
        if obj.processed == True:
            continue
        else:
            process(objects1, objects2, rtree1, rtree2, obj, pq, dest_objects)
    """
    while pq.qsize() != 0:
        q = pq.get()
        process(dict_objects1, dict_objects2, rtree1, rtree2, q, pq, dest_objects)
    return

def merge(optics_instance1, optics_instance2):
    #objects1 = optics_instance1.get_optics_objects()
    #objects2 = optics_instance2.get_optics_objects()
    objects1 = optics_instance1
    objects2 = optics_instance2
    #co1 = optics_instance1.get_cluster_ordering()
    #co2 = optics_instance2.get_cluster_ordering()
    
    property1 = index.Property()
    idx1 = index.Index(properties=property1)
    for obj in objects1:
        idx1.insert(obj.index_object, obj.point)

    property2 = index.Property()
    idx2 = index.Index(properties=property2)
    for obj in objects2:
        idx2.insert(obj.index_object, obj.point)

    dict_objects1 = {}
    dict_objects2 = {}
    for obj in objects1:
        dict_objects1[obj.index_object] = obj
    for obj in objects2:
        dict_objects2[obj.index_object] = obj

    mark_affected_points(dict_objects1, dict_objects2, idx1, idx2)

    dest_objects = []
    process_co(dict_objects1, dict_objects2, idx1, idx2, dest_objects)
    process_co(dict_objects2, dict_objects1, idx1, idx2, dest_objects)

    dest_objects1 = list(dict_objects1.values())
    dest_objects2 = list(dict_objects2.values())

    for obj in dest_objects2:
        #obj.index_object = (obj.index_object + 1) * len(dest_objects1)
        obj.index_object = obj.index_object + len(dest_objects1)

    dest_objects = dest_objects1
    dest_objects.extend(dest_objects2)

    return dest_objects

def reset_flags(optics_objects):
    for obj in optics_objects:
        obj.processed = False
        obj.affected = False

def mpi_optics(input_filepath, eps, minpts):
    if comm.rank == 0:
        data = np.loadtxt(input_filepath, delimiter=",").tolist()
        tree = kdtree.create(data)
        root_height = tree.height()
        dest_height = int(np.log2(comm.size))
        dest_nodes = [node for node in tree.inorder() if root_height - node.height() == dest_height]
        divided_data = [[node.data for node in list(node.inorder())] for node in dest_nodes]
        divided_data = zip(list(divided_data), range(0, size))
        for data in divided_data:
            dest = data[1]
            data = data[0]
            if dest != 0:
                comm.send(data, dest=dest, tag=dest)
            else:
                distributed_data = data
    else:
        distributed_data = comm.recv(source=0, tag=rank)

    print("rank=%s, size=%s, name=%s" % (rank, len(distributed_data), name))

    optics_instance = optics(distributed_data, eps, minpts)
    optics_instance.process()
    print("OPTICS finished! rank = %s" % rank)

    optics_instance = optics_instance.get_optics_objects()

    ite_num = int(np.log2(comm.size))
    for ite in range(0, ite_num):
        psuedo_id = int(rank / pow(2, ite))
        if psuedo_id % 2 == 1:
            dest = int((psuedo_id - 1) * pow(2, ite))
            comm.send(optics_instance, dest=dest, tag=ite*10+dest)
            #print("sent dest = %s rank = %s ite = %s" % (dest, rank, ite))
            break
        else:
            source = int((psuedo_id + 1) * pow(2, ite))
            received = comm.recv(source=source, tag=ite*10+rank)
            reset_flags(optics_instance)
            reset_flags(received)
            optics_instance = merge(optics_instance, received)
            #print("received = %s rank = %s ite = %s" % (received, rank, ite))
    
    if rank == 0:
        #dataset = distributed_data
        #clusters = optics_instance.get_clusters()
        #noise = optics_instance.get_noise()
        #cluster_ordering = optics_instance.get_cluster_ordering()
        #print(cluster_ordering)
        optics_objects = optics_instance
        #visualizer = cluster_visualizer();
        #visualizer.append_clusters(clusters, dataset);
        #visualizer.append_cluster(noise, dataset, marker = 'x');
        #visualizer.show();
        print("FINAL LEN = %s" % len(optics_objects))

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 4:
        sys.exit("Usage: mpirun -np <processor_num> python %s <filename> <eps> <minpts>" % sys.argv[0])
    input_filepath = argv[1]
    eps = float(argv[2])
    minpts = int(argv[3])

    if rank == 0:
        start = time.time()

    mpi_optics(input_filepath, eps, minpts)

    if rank == 0:
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")