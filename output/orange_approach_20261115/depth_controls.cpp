#include "libobsensor/ObSensor.hpp"
#include <iostream>
int main() try {
 ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
 ob::Context ctx;auto list=ctx.queryDeviceList();if(!list->deviceCount())return 2;auto dev=list->getDevice(0);
 for(auto id:{OB_PROP_IR_EXPOSURE_INT,OB_PROP_IR_GAIN_INT,OB_PROP_MIN_DEPTH_INT,OB_PROP_MAX_DEPTH_INT,
              OB_PROP_LASER_POWER_LEVEL_CONTROL_INT,OB_PROP_LASER_POWER_ACTUAL_LEVEL_INT})try{
  bool readable=dev->isPropertySupported(id,OB_PERMISSION_READ);std::cout<<"INT_PROP "<<id<<" readable "<<readable;
  if(readable){auto r=dev->getIntPropertyRange(id);std::cout<<" current "<<r.cur<<" min "<<r.min<<" max "<<r.max<<" default "<<r.def;}
  std::cout<<std::endl;
 }catch(ob::Error&e){std::cout<<"PROPERTY_ERROR "<<id<<" "<<e.getMessage()<<std::endl;}
 for(auto id:{OB_PROP_IR_AUTO_EXPOSURE_BOOL,OB_PROP_LASER_BOOL,OB_PROP_DEPTH_SOFT_FILTER_BOOL,OB_PROP_DEPTH_HOLEFILTER_BOOL})try{
  if(dev->isPropertySupported(id,OB_PERMISSION_READ))std::cout<<"BOOL_PROP "<<id<<" value "<<dev->getBoolProperty(id)<<std::endl;
 }catch(ob::Error&e){std::cout<<"PROPERTY_ERROR "<<id<<" "<<e.getMessage()<<std::endl;}
 return 0;
}catch(ob::Error&e){std::cerr<<e.getMessage()<<std::endl;return 1;}
