#include <mbot_object_localization_ros/object_localization.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "object_localizer");

    // create object of the node class (ObjectLocalization)
    object_2d_to_3d::ObjectLocalization object_localization;

    ROS_INFO("Node initialized");

    //main loop function
    object_localization.loop();

    return 0;
}
