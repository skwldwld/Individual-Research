import open3d as o3d

# 예시 점군 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

# 시각화
o3d.visualization.draw_geometries([pcd])
