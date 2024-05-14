import open3d as o3d

mesh = o3d.io.read_triangle_mesh('models/2005 toyota - Salin.obj')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
