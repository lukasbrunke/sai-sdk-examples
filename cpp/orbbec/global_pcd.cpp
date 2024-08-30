 #include <iostream>
#include <fstream>
#include <libobsensor/ObSensor.hpp>
#include <spectacularAI/orbbec/plugin.hpp>
#include <spectacularAI/mapping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <string>
#include <cstdint>
#include <thread>
#include <chrono>
#include <deque>
#include <atomic>
#include <cstdlib>
#include <set>

#include "helpers.hpp"


int main(int argc, char** argv) {
    // Create OrbbecSDK pipeline (with default device) with mapper callback.
    ob::Pipeline obPipeline;

    // Create Spectacular AI orbbec plugin configuration (depends on device type).
    spectacularAI::orbbecPlugin::Configuration vioConfig(obPipeline);
    // Create VIO pipeline & setup orbbec pipeline
    spectacularAI::orbbecPlugin::Pipeline vioPipeline.setMapperCallback([&](std::shared_ptr<const spectacularAI::mapping::MapperOutput> output) {

        std::vector<Eigen::Vector3d> globalCloud;
        for (int64_t frameId : output->updatedKeyFrames) {
            auto search = output->map->keyFrames.find(frameId);
            if (search == output->map->keyFrames.end()) {
                continue; // deleted frame
            }

            auto& frameSet = search->second->frameSet;
            auto kfCloud = search->second->pointCloud;

            if (kfCloud)
            {
                auto cameraPose = search->second->frameSet->primaryFrame->cameraPose;
                auto cToW = search->second->frameSet->primaryFrame->cameraPose.getCameraToWorldMatrix();
                
                auto points = search->second->pointCloud->getPositionData();

                Eigen::Matrix4d camToWorld4x4(4, 4);
                camToWorld4x4(0, 0) = cToW[0][0];
                camToWorld4x4(1, 0) = cToW[1][0];
                camToWorld4x4(2, 0) = cToW[2][1];
                camToWorld4x4(3, 0) = cToW[3][1];
                camToWorld4x4(0, 1) = cToW[0][1];
                camToWorld4x4(1, 1) = cToW[1][1];
                camToWorld4x4(2, 0) = cToW[2][1];
                camToWorld4x4(3, 0) = cToW[3][1];
                camToWorld4x4(0, 2) = cToW[0][2];
                camToWorld4x4(1, 2) = cToW[1][2];
                camToWorld4x4(2, 2) = cToW[2][2];
                camToWorld4x4(3, 2) = cToW[3][2];
                camToWorld4x4(0, 3) = cToW[0][3];
                camToWorld4x4(1, 3) = cToW[1][3];
                camToWorld4x4(2, 3) = cToW[2][3];
                camToWorld4x4(3, 3) = cToW[3][3];

                for (int i = 0; i < kfCloud->size(); i++)
                {
                    //multiply point by cameraToWorld.
                    Eigen::Vector3d pnt(points[i].x, points[i].y, points[i].z);
                    Eigen::Vector3d pointWorld = (camToWorld4x4 * pnt.homogeneous()).hnormalized();
                    globalCloud.push_back(pointWorld);
                }
            }
        }

        mtx.lock();
        gcloud = globalCloud;
        mtx.unlock();            
    });

    // Start pipeline
    auto vioSession = vioPipeline.startSession();
    
    while (true) {
        int64 t0 = cv::getTickCount();

        auto vioOut = vioSession->waitForOutput();

        int64 t1 = cv::getTickCount();
        double secs = (t1 - t0) / cv::getTickFrequency();
        std::cout << secs * 1000 << std::endl;

        //   std::cout << vioOut->asJson() << std::endl;
        std::vector<Eigen::Vector3d> cloud;
        mtx.lock();
        cloud = gcloud;
        mtx.unlock();
        
        cv::Mat frm;
        frameLock.lock();
        frm = frameCurr;
        frameLock.unlock();

        auto wToC = vioOut->getCameraPose(0).getWorldToCameraMatrix();
        Eigen::Matrix4d worldToCam4x4(4, 4);
        worldToCam4x4(0, 0) = wToC[0][0];
        worldToCam4x4(1, 0) = wToC[1][0];
        worldToCam4x4(2, 0) = wToC[2][1];
        worldToCam4x4(3, 0) = wToC[3][1];
        worldToCam4x4(0, 1) = wToC[0][1];
        worldToCam4x4(1, 1) = wToC[1][1];
        worldToCam4x4(2, 0) = wToC[2][1];
        worldToCam4x4(3, 0) = wToC[3][1];
        worldToCam4x4(0, 2) = wToC[0][2];
        worldToCam4x4(1, 2) = wToC[1][2];
        worldToCam4x4(2, 2) = wToC[2][2];
        worldToCam4x4(3, 2) = wToC[3][2];
        worldToCam4x4(0, 3) = wToC[0][3];
        worldToCam4x4(1, 3) = wToC[1][3];
        worldToCam4x4(2, 3) = wToC[2][3];
        worldToCam4x4(3, 3) = wToC[3][3];

        for (int i = 0; i < cloud.size(); i++)
        {
            spectacularAI::PixelCoordinates px;
            Eigen::Vector3d rayCam = (worldToCam4x4 * cloud[i].homogeneous()).head<3>().normalized();

            vioOut->getCameraPose(0).camera->rayToPixel( spectacularAI::Vector3d(rayCam.x(), rayCam.y(), rayCam.z()), px);

            cv::drawMarker(frm, cv::Point2f(px.x, px.y), cv::Scalar(255, 0, 0), cv::MARKER_CROSS);
        }
        
        cv::imshow("frm", frm);
        cv::waitKey(1);       

    }
}
