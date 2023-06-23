import cv2
import numpy as np 
import open3d as o3d

def image_vis(image_t):
    image_np = image_t.data.permute(1,2,0).cpu().numpy()
    image_np = (255*image_np).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./pictures/image_np.png', image_np)
    cv2.imshow('image_np', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def depth_vis(depth_t):
    # Convert to numpy
    depth_np = depth_t.permute(1,2,0).squeeze().data.cpu().numpy()
    # Normalize
    depth_np = depth_np - depth_np.min()
    depth_np = depth_np / (depth_np.max()-depth_np.min())
    depth_np = (255 * depth_np).astype(np.uint8)
    # Color visualization
    depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_PLASMA)
    cv2.imshow('depth_color', depth_color)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def depth_vis_for_test_simple(depth_t):
    # Convert to numpy
    depth_np = depth_t.permute(1,2,0).squeeze().data.cpu().numpy()
    # Normalize
    depth_np = depth_np - depth_np.min()
    depth_np = depth_np / (depth_np.max()-depth_np.min())
    depth_np = (255 * depth_np).astype(np.uint8)
    # Color visualization
    depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_PLASMA)
    cv2.imshow('depth_color', depth_color)
    cv2.waitKey(10)
    # cv2.destroyAllWindows()
    
def pcd_vis(pcd, show_origin=True, origin_size=3, show_grid=True):
    # # Convert to numpy
    # pcd = pcd[0].permute(1,0)[:,0:3].data.cpu().numpy()
    # # Normalize
    # pcd = pcd - pcd.min()
    # # pcd = pcd / pcd.max()
    # pcd = pcd / (pcd.max() - pcd.min())
    # pcd = (255 * pcd).astype(np.uint8)
    
    cloud = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    
    if isinstance(pcd, type(cloud)):
        pass
    elif isinstance(pcd, np.ndarray):
        cloud.points = v3d(pcd)
        
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))
    
    # set front, lookat, up, zoom to change initial view
    return o3d.visualization.draw_geometries([cloud, coord])
    
    
