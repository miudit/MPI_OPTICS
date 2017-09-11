import sys

from pyclustering.cluster.optics import optics
from pyclustering.cluster import cluster_visualizer
import kdtree

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

def mpi_optics(input_filepath, eps, minpts):
    if comm.rank == 0:
        data = np.loadtxt(input_filepath, delimiter=",").tolist()
        tree = kdtree.create(data)
        root_height = tree.height()
        dest_height = int(np.log2(comm.size))
        dest_nodes = [node for node in tree.inorder() if root_height - node.height() == dest_height]
        divided_data = [[node.data for node in list(node.inorder())] for node in dest_nodes]
    else:
        divided_data = None

    data = comm.scatter(divided_data, root=0)

    optics_instance = optics(data, eps, minpts)
    optics_instance.process()
    clusters = optics_instance.get_clusters()
    noise = optics_instance.get_noise()

    print(clusters)

    visualizer = cluster_visualizer();
    visualizer.append_clusters(clusters, data);
    visualizer.append_cluster(noise, data, marker = 'x');
#    visualizer.show();

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 4:
        sys.exit("Usage: mpirun -np <processor_num> python %s <filename> <eps> <minpts>" % sys.argv[0])
    input_filepath = argv[1]
    eps = float(argv[2])
    minpts = int(argv[3])
    mpi_optics(input_filepath, eps, minpts)
