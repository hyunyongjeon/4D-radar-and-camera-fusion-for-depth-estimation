import open3d as o3d
from multiprocessing import Process, Queue

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def o3dvis(video):
    print("o3dvis process ...")
    o3dvis.points = []

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        # if len(o3dvis.points) > 0:
        #     for point_actor in o3dvis.points:
        #         print(point_actor)
        #         vis.remove_geometry(point_actor)
        #         o3dvis.points.pop(0)

        points, colors = [], []
        while not video.pcd.empty():
            point, color = video.pcd.get()
            points.append(point)
            colors.append(color)
        for i in range(len(points)):
            pts = points[i][::1]
            clr = colors[i][::1]
            point_actor = create_point_actor(pts, clr)
            
            vis.clear_geometries()
            vis.add_geometry(point_actor)

            o3dvis.points.append(point_actor)

        # hack to allow interacting with vizualization during inference
        cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        vis.poll_events()
        vis.update_renderer()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)

    vis.create_window(height=540, width=960)

    vis.run()
    vis.destroy_window()


class pcdBuffer():
    def __init__(self):
        self.pcd = Queue()

