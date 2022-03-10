import CameraIface

# create an instance of the camera interface class
myCameraIface = CameraIface.CameraIface(3)

# call the class method to calibrate the camera interface
myCameraIface.calibrate()

# then continuously print the position e.g. which level
for _ in range(10000000000000000000):
    print(myCameraIface.get_object_level())
    print(myCameraIface.get_object_position())
