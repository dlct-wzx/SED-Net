import os
import numpy as np
import open3d as o3d



for i in range(10):
    with open(os.path.join('results',str(i) + '_edge.txt')) as f:
        edges = f.readlines()
        edges = [line.split(';') for line in edges]
        edges = [[float(x) for x in line] for line in edges]
        edges = np.array(edges)

    with open('results/' + str(i) + '_Vis_inst.txt') as f:
        xyz = f.readlines()
        xyz = [line.split(';') for line in xyz]
        xyz = [[float(x) for x in line] for line in xyz]
        xyz = np.array(xyz)


    red = [255, 0, 0]
    black = [0, 0, 0]

    # edge 中0 > 1则为红色，否则为黑色
    colors = [red if x[0] < x[1] else black for x in edges]
    xyz = xyz[:, :3]
    xyz = np.concatenate((xyz, colors), axis=1)
    print(xyz.shape)
    # 保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyz[:, 3:])

    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("0.ply", pcd)