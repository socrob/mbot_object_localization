//
// Created by tdias on 07-06-2018.
//

#ifndef MBOT_OBJECT_LOCALIZATION_OBJECT_LOCALIZATION_NODE_H
#define MBOT_OBJECT_LOCALIZATION_OBJECT_LOCALIZATION_NODE_H

#include <ros/ros.h>
#include <ros/console.h>

#include <string>
#include <vector>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <mbot_perception_msgs/RecognizedObject3D.h>
#include <mbot_perception_msgs/RecognizedObject3DList.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <darknet_ros_py/RecognizedObjectArrayStamped.h>
#include <darknet_ros_py/RecognizedObject.h>
#include <geometry_msgs/PoseArray.h>

//  TODO remove unused includes
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Camera info and geometry
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>


#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

namespace object_2d_to_3d {
    using namespace message_filters::sync_policies;


    struct DetectedObject {
        const std::string class_name;
        const float confidence;
        const cv::Rect original_roi;
        const cv::Mat roi_mat;
        cv::Mat intersection_mask;

        DetectedObject(const std::string _class, const float _confidence, const cv::Rect&& _roi, const cv::Mat&& _original_mat ) :
          class_name(_class), confidence(_confidence), original_roi(_roi), roi_mat(_original_mat, _roi),
          intersection_mask(cv::Mat::zeros(_original_mat.size(), _original_mat.type())){
            // nothing to do
          }
    };


    class ObjectLocalization {
        typedef ApproximateTime<sensor_msgs::Image, darknet_ros_py::RecognizedObjectArrayStamped> DepthToBBSyncPolicy;

    public:

        // default constructor
        ObjectLocalization();

        // destructor
        ~ObjectLocalization();

        // publisher -> subscriber connect callback
        void connectCallback();

        // publisher -> subscriber disconnect callback
        void disconnectCallback();

        // listens to camera info topic to get the intrinsic parameters
        void cameraCallback(sensor_msgs::CameraInfoConstPtr info);

        // check if there is at least one subscriber to the output topics
        bool thereAreSubscribersToOutput();

        // sync bounding boxes to depth map
        void
        syncCallback(const sensor_msgs::ImageConstPtr &depth, const darknet_ros_py::RecognizedObjectArrayStampedConstPtr &rec);

        void loop();

    private:

        void handleArray(const darknet_ros_py::RecognizedObjectArrayStamped &rec) const;

        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        ros::Subscriber cam_info_sub_;

        sensor_msgs::Image last_depth_received_;
        image_transport::Publisher result_img_pub_;
        message_filters::Subscriber<sensor_msgs::Image> depth_img_sub_;

        message_filters::Subscriber<darknet_ros_py::RecognizedObjectArrayStamped> object_array_sub_;
        ros::Publisher object_pub_;
        ros::Publisher event_out_pub_;
        ros::Publisher poses_pub_;

        // synchronizer of BB and depth
        std::unique_ptr<message_filters::Synchronizer<DepthToBBSyncPolicy>> sync_;

        // variables for storing the yaml parameters
        float width_fov_;
        float height_fov_;
        bool synchronize_frames_;

        // pinhole camera model
        image_geometry::PinholeCameraModel model_;

        double inner_ratio_;
        bool center_inner_region_ = true;
        bool cam_info_received_{false};
        bool depth_image_received_{false};
        bool roi_received_{false};
        bool filter_classes_enable_{false};
        std::vector<std::string> filter_classes_;
        bool remove_intersections_{false};

        cv_bridge::CvImageConstPtr last_received_depth_;
        darknet_ros_py::RecognizedObjectArrayStamped last_received_array_;

        void createSubscribers();
        void destroySubscribers();
    };
} //end of namespace
#endif //MBOT_OBJECT_LOCALIZATION_OBJECT_LOCALIZATION_NODE_H

