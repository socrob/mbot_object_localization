//
// Created by tdias on 07-06-2018.
//

#include <mbot_object_localization_ros/object_localization.h>
#include <algorithm>
#include <memory>

namespace object_2d_to_3d {

    constexpr float NAN_ALLOWED_RATIO = 0.25;
    constexpr float DONT_PUBLISH = -1;

    float getDepthBySorting(const DetectedObject& object, const float inner_ratio, const bool center_inner_region)
    {
        cv::Mat flat;
        cv::Mat flat_ord;

        const cv::Mat& mat = object.roi_mat;

        const int inner_width = ceil(inner_ratio * mat.cols);
        const int inner_height = ceil(inner_ratio * mat.rows);
        cv::Rect inner;

        if (center_inner_region){
            // position the inner region in the center of the boundingbox roi
            inner = cv::Rect(cv::Point2i(mat.cols/2, mat.rows/2) - cv::Point2i(inner_width/2.0, inner_height/2.0), cv::Size(inner_width, inner_height));
        } else {
            // position the inner region at the bottom of the boundingbox roi
            inner = cv::Rect(cv::Point2i(mat.cols/2, mat.rows) - cv::Point2i(inner_width/2.0, inner_height), cv::Size(inner_width, inner_height));
        }

        // Needs copy because it is not continuous
        const cv::Mat inner_mat(mat, inner);
        // Check NANs
        unsigned int nan_count=0, intersection_count=0;
        std::vector<float> inner_filtered;
        for(int row=0; row < inner_mat.rows; ++row)
        {
            for(int col=0; col < inner_mat.cols; ++col)
            {
                if(object.intersection_mask.at<int>(row + object.original_roi.y + inner.y, col + object.original_roi.x + inner.x) == 0)
                {
                    intersection_count++;
                }
                else
                {
                    const auto val = inner_mat.at<float>(row,col);
                    if(std::isnan(val))
                        nan_count++;
                    else
                        inner_filtered.push_back(val);
                }
            }
        }

        // If number of NANs is too high, return -1 to signal no-publish
        const unsigned int num_values = inner_mat.rows * inner_mat.cols;

        if(nan_count+intersection_count > (1-NAN_ALLOWED_RATIO)*num_values)
        {
            ROS_DEBUG("Object: %s - Too many NANs (%d NAN and %d intersection out of %d values) in the provided bounding-boxed depth map, do not publish", object.class_name.c_str(), nan_count, intersection_count, num_values);
            return DONT_PUBLISH;
        }

        if(inner_filtered.empty())
        {
            ROS_DEBUG("Object: %s - NAN-Filtered vector is empty, try to see if the bounding box matches the depthmap.", object.class_name.c_str());
            return DONT_PUBLISH;
        }

        // Faster way to get the 25th percentile
        std::nth_element(inner_filtered.begin(), inner_filtered.begin() + inner_filtered.size()/4, inner_filtered.end());
        return inner_filtered[inner_filtered.size()/4];
    }

    float getDepthByCenter(const cv::Mat& mat)
    {
        return mat.at<float>(mat.size()/2);
    }

    ObjectLocalization::ObjectLocalization() : nh_("~"), it_(nh_)
    {
        // get params
        nh_.getParam("width_fov", width_fov_);
        nh_.getParam("height_fov", height_fov_);
        nh_.getParam("synchronize_frames", synchronize_frames_);
        nh_.getParam("inner_ratio", inner_ratio_);
        nh_.getParam("center_inner_region", center_inner_region_);
        nh_.param<bool>("remove_intersections", remove_intersections_, false);
        nh_.getParam("filter_classes", filter_classes_);

        if(filter_classes_.empty())
        {
            filter_classes_enable_ = false;
            ROS_INFO("All classes will be localized");
        }
        else
        {
            filter_classes_enable_ = true;
            ROS_INFO("Will not consider any class that is not one of the following: ");
            for(const auto& cl: filter_classes_)
                ROS_INFO_STREAM("\t\t\t\t\t" << cl);
        }

        // subscriber events setup
        ros::SubscriberStatusCallback connect_cb = boost::bind(&ObjectLocalization::connectCallback, this);
        ros::SubscriberStatusCallback disconnect_cb = boost::bind(&ObjectLocalization::disconnectCallback, this);

        //output event
        event_out_pub_ = nh_.advertise<std_msgs::String>("event_out", 5);

        //output publisher
        object_pub_ = nh_.advertise<mbot_perception_msgs::RecognizedObject3DList>("localized_objects", 5, connect_cb, disconnect_cb);

        //debug poses publisher
        poses_pub_ = nh_.advertise<geometry_msgs::PoseArray>("localized_object_poses", 5, connect_cb, disconnect_cb);

        //subscribe to camera info
        cam_info_sub_ = nh_.subscribe("/camera/depth/camera_info", 1, &ObjectLocalization::cameraCallback, this);

    }

    ObjectLocalization::~ObjectLocalization() {
        destroySubscribers();
    }

    void ObjectLocalization::connectCallback() {

        if( !sync_ ) {
            createSubscribers();
            ROS_INFO("Created subscribers because there are connections on output topics");
        }
    }

    void ObjectLocalization::disconnectCallback() {

        if( !thereAreSubscribersToOutput() && sync_ ) {
            destroySubscribers();
            ROS_INFO("Destroyed subscribers because there are no connections on output topics");
        }
    }

    void ObjectLocalization::handleArray(const darknet_ros_py::RecognizedObjectArrayStamped &rec) const {

        const cv::Mat depmat = last_received_depth_->image;
        //const int depth_frame_width = depmat.size().width;
        //const int depth_frame_height = depmat.size().height;

        const unsigned long nrObjects = rec.objects.objects.size();

        std_msgs::String e_out;

        mbot_perception_msgs::RecognizedObject3DList object_3d_list;
        object_3d_list.image_header = rec.header;
        object_3d_list.header.frame_id = last_received_depth_->header.frame_id;
        object_3d_list.header.stamp = ros::Time::now();

        if (nrObjects == 0)
        {
            ROS_INFO("There are no objects");
            e_out.data = "e_no_objects";
            event_out_pub_.publish(e_out);

            // publish empty anyway
            object_pub_.publish(object_3d_list);
            geometry_msgs::PoseArray pose_arr;
            pose_arr.header.frame_id = last_received_depth_->header.frame_id;
            pose_arr.header.stamp = ros::Time::now();
            poses_pub_.publish(pose_arr);
            return;
        }


        mbot_perception_msgs::RecognizedObject3D object_3d;
        object_3d.pose.orientation.w = 1.0;

        std::vector<DetectedObject> detections;

        // Filter the objects
        for(const darknet_ros_py::RecognizedObject& object : rec.objects.objects)
        {
            if(filter_classes_enable_) {
                if (std::find(filter_classes_.begin(), filter_classes_.end(), object.class_name) == filter_classes_.end())
                    continue;
            }

            const sensor_msgs::RegionOfInterest& roi = object.bounding_box;

            try
            {
                // obtain the image ROI:
                depmat.convertTo(depmat, CV_32F, 1.0);
            } catch (const cv::Exception& ex)
            {
                ROS_ERROR("%s",ex.what());
                return;
            }

            // Put the bounding box in the list
            detections.emplace_back(
                DetectedObject(object.class_name, object.confidence,
                               cv::Rect(roi.x_offset, roi.y_offset, roi.width, roi.height),
                               cv::Mat(depmat))
            );
        }

        // If enabled, remove intersections with other objects' bounding boxes
        for(auto it = detections.begin(); it != detections.end(); ++it)
        {
            // Initialize mask with roi
            it->intersection_mask(it->original_roi) = 1;

            // If removing intersections is not enabled, stop here
            if(false == remove_intersections_)
                continue;

            // Loop through all other detections and check for intersections
            for(auto it2 = detections.begin(); it2 != detections.end(); ++it2)
            {
                if(it == it2)
                    continue;

                // Intersection operator is simply &, put those values as zero in the mask
                it->intersection_mask(it->original_roi & it2->original_roi) = 0;
            }
        }

        for(const auto& object: detections)
        {
            float depth = getDepthBySorting(object, inner_ratio_, center_inner_region_);

            if(depth == DONT_PUBLISH || std::isnan(depth))
            {
                std_msgs::String e_out;
                e_out.data = "e_nans";
                event_out_pub_.publish(e_out);
                continue;
            }

            // center of bounding box in image frame
            const cv::Point2d center(0.5*(object.original_roi.br()+object.original_roi.tl()));

            // unit vector in camera coordinates
            const cv::Point3d unit_3d = model_.projectPixelTo3dRay(center);

            // multiply by distance to obtain 3d point
            object_3d.pose.position.x = unit_3d.x * depth;
            object_3d.pose.position.y = unit_3d.y * depth;
            object_3d.pose.position.z = unit_3d.z * depth;

            object_3d.class_name = object.class_name;
            object_3d.confidence = object.confidence;

            object_3d_list.objects.push_back(object_3d);
        }
        // publishing object poses
        object_pub_.publish(object_3d_list);

        // publishing output event
        e_out.data = "e_published";
        event_out_pub_.publish(e_out);

        // if someone subscribed to debug topic, publish it
        geometry_msgs::PoseArray pose_arr;
        pose_arr.header = object_3d_list.header;

        for(const auto& obj: object_3d_list.objects)
        {
            pose_arr.poses.push_back(obj.pose);
        }

        poses_pub_.publish(pose_arr);
    }

    bool ObjectLocalization::thereAreSubscribersToOutput() {
        
        return (object_pub_.getNumSubscribers() > 0 || poses_pub_.getNumSubscribers() > 0); 
    }

    void ObjectLocalization::destroySubscribers() {

        sync_.reset(nullptr);
        depth_img_sub_.unsubscribe();
        object_array_sub_.unsubscribe();
    }

    void ObjectLocalization::createSubscribers() {

        depth_img_sub_.subscribe(nh_, "/camera/depth/image_rect", 1);
        object_array_sub_.subscribe(nh_, "/detection_result", 1);
        sync_.reset(new message_filters::Synchronizer<DepthToBBSyncPolicy> (
                DepthToBBSyncPolicy(10), depth_img_sub_, object_array_sub_ ));
        sync_->registerCallback(
                boost::bind(&ObjectLocalization::syncCallback, this, _1, _2)
        );
    }

    void ObjectLocalization::loop() {
        double node_frequency;
        nh_.param<double>("node_frequency", node_frequency, 50.0);
        ROS_INFO("Node will run at : %lf [hz]", node_frequency);

        ros::Rate loop_rate(node_frequency);
        while (ros::ok()) {

            if (cam_info_received_ && depth_image_received_ && roi_received_) {
                handleArray(last_received_array_);
                depth_image_received_ = false;
                roi_received_ = false;
            }

            //listen to callbacks
            ros::spinOnce();

            loop_rate.sleep();
        }
    }

    void ObjectLocalization::syncCallback(const sensor_msgs::ImageConstPtr &depth,
                                          const darknet_ros_py::RecognizedObjectArrayStampedConstPtr &rec) {

        // If no objects, don't do anything
        if(rec->objects.objects.empty())
        {
            std_msgs::String e_out;
            e_out.data = "e_no_objects";
            event_out_pub_.publish(e_out);
        
            mbot_perception_msgs::RecognizedObject3DList object_3d_list;
            object_3d_list.header = rec->header;
            object_3d_list.image_header = depth->header;
            object_pub_.publish(object_3d_list);

            geometry_msgs::PoseArray pose_arr;
            pose_arr.header = rec->header;
            poses_pub_.publish(pose_arr);
            return;
        }

        // Save ROI
        if(filter_classes_enable_)
        {
            bool any_obj_of_class = false;
            for(const auto& obj: rec->objects.objects)
            {
                if(std::find(filter_classes_.begin(), filter_classes_.end(), obj.class_name) != filter_classes_.end()) {
                    any_obj_of_class = true;
                    break;
                }
            }

            // Dont save objects if none is of desired class
            if(!any_obj_of_class) {
                std_msgs::String e_out;
                e_out.data = "e_no_objects_of_class";
                event_out_pub_.publish(e_out);

                // Publish empty anyway
                mbot_perception_msgs::RecognizedObject3DList object_3d_list;
                object_3d_list.header = rec->header;
                object_3d_list.image_header = depth->header;
                object_pub_.publish(object_3d_list);
                
                geometry_msgs::PoseArray pose_arr;
                pose_arr.header = rec->header;
                poses_pub_.publish(pose_arr);
                return;
            }
        }

        last_received_array_ = *rec;
        roi_received_ = true;

        // Save depth
        try {
            last_received_depth_ = cv_bridge::toCvCopy(*depth);
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("Could not convert from encoding '%s'.", depth->encoding.c_str());
            return;
        }
        depth_image_received_ = true;
    }

    void ObjectLocalization::cameraCallback(sensor_msgs::CameraInfoConstPtr info)
    {
        cam_info_received_ = true;
        ROS_INFO("Camera info received");

        if(!model_.fromCameraInfo(*info))
        {
            ROS_FATAL("Error saving camera model");
            exit(-1);
        }

        std_msgs::String e_out;
        e_out.data = "e_cam_info_received";
        event_out_pub_.publish(e_out);
        cam_info_sub_.shutdown();
    }
} // namespace object_2d_to_3d
