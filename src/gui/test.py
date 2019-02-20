import vtk
import cv2
import numpy as np
from vtk.util.colors import tomato
from xml.etree.ElementTree import tostring




image = cv2.imread("/home/oscar/eclipse-workspace/HistopathologicalCharacterization/input/map.png")

print image.shape
 
image = image.astype(np.uint8) 

#cv2.imshow("",image)
#cv2.waitKey(0)

stringImage = image.tostring()

dataImporter = vtk.vtkImageImport()

dataImporter.CopyImportVoidPointer(stringImage, len(stringImage))
dataImporter.SetDataScalarTypeToUnsignedChar()
dataImporter.SetWholeExtent(0, image.shape[1], 0, image.shape[0], 0, image.shape[2])
dataImporter.SetDataExtent(0, image.shape[1], 0, image.shape[0], 0, image.shape[2])

compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
# This class describes how the volume is rendered (through ray tracing).
compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
# We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
volumeMapper = vtk.vtkVolumeRayCastMapper()
volumeMapper.SetVolumeRayCastFunction(compositeFunction)
volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
#volumeProperty.SetScalarOpacity(alphaChannelFunc)


# The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
renderer = vtk.vtkRenderer()
renderWin = vtk.vtkRenderWindow()
renderWin.AddRenderer(renderer)
renderInteractor = vtk.vtkRenderWindowInteractor()
renderInteractor.SetRenderWindow(renderWin)

# We add the volume to the renderer ...
renderer.AddVolume(volume)
# ... set background color to white ...
renderer.SetBackground(255,255,255)
# ... and set window size.
renderWin.SetSize(image.shape[1], image.shape[0])
# Tell the application to use the function as an exit check.


renderInteractor.Initialize()
# Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
renderWin.Render()
renderInteractor.Start()



