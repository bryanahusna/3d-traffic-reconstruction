import open3d as o3d

class CarVoxelMesh:
  def __init__(self, filename: str):
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.1)
    print(voxel_grid)
    print(dir(voxel_grid))
    #o3d.visualization.draw_geometries([voxel_grid])
