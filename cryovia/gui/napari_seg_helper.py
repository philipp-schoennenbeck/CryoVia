"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from PyQt5 import QtGui
from scipy.fft import fft2, ifft2
from magicgui import magic_factory
from scipy.ndimage import label

# from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLineEdit, QFileDialog, QCheckBox, QLabel, QGridLayout, QMessageBox, QShortcut, QDialog,QDialogButtonBox
from PyQt5.QtGui import QDoubleValidator
import mrcfile
import numpy as np
from napari import Viewer
import sparse
from pathlib import Path
# from napari_mrcfile_reader.mrcfile_reader import reader_function
import mrcfile

from cryovia.gui.segmentation_files.prep_training_data import fft_rescale_image, load_file
if TYPE_CHECKING:
    import napari

def mrc_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, (str,Path)) else path
    # load all files into array
    try:

        arrays = [mrcfile.open(_path).data for _path in paths]
    except:
        arrays = [mrcfile.open(_path, permissive=True).data for _path in paths]
     
    # stack arrays into single array
    # data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    # https://napari.org/docs/api/napari.components.html#module-napari.components.add_layers_mixin

    voxel_sizes = [mrcfile.open(path,permissive=True,header_only=True).voxel_size.x for path in paths]
    # voxel_size = header_data.voxel_size.x
    
    add_kwargs = [{"metadata":{"pixel_spacing":voxel_size}} for voxel_size in voxel_sizes]
    layer_types = ["labels" if data[0].dtype.char in np.typecodes["AllInteger"] else "image" for data in arrays  ]
    # if data[0].dtype.char in np.typecodes["AllInteger"]:
    #     layer_type = "labels"
    # else:
    #     layer_type = "image"  # optional, default is "image"
    return [(data, add_kwarg, layer_type) for data, add_kwarg, layer_type in  zip(arrays, add_kwargs, layer_types)]


class DefaultFloatValidator(QDoubleValidator):
    def __init__(self, default):
        super().__init__()
        self.default = default
    def fixup(self, a0: str) -> str:
        if len(a0) == 0:
            return str(self.default)
        if float(a0) <= 0:
            return str(self.default)
        return super().fixup(a0)


class SegmentationHelper(QWidget):
    def __init__(self, viewer:Viewer, savedir=Path().cwd(), files=[], custom_parent=None, segmentation_model=None, default_pixel_size=7):
        super().__init__()
        self.viewer:Viewer = viewer
        self.custom_parent = custom_parent
        self.segmentationModel = segmentation_model
        self.files = files
        if self.segmentationModel is not None:
            self.pixelSize = self.segmentationModel.config.pixel_size
        else:
            self.pixelSize = default_pixel_size

        self.nonMrcPixelSize = default_pixel_size
        if self.segmentationModel is not None:
            savedir = Path().cwd() / segmentation_model.name
            savedir.mkdir(parents=True, exist_ok=True)
        self.idx = -1
        self.askAgain = True
        self.answer = True

        self.finishedSegmentations = [[],[]]

        self.setLayout(QVBoxLayout())

        self.loadFilesButton = QPushButton("Load/Add files")
        self.loadFilesButton.clicked.connect(self.loadFiles)

        self.chooseSaveDirButton = QPushButton("Choose save directory")
        self.chooseSaveDirButton.clicked.connect(self.chooseSaveDir)
        

        self.saveDirLabel = QLabel("")
        self.saveDir = savedir
        

        self.upButton = QPushButton("Next membrane (w)")
        self.upButton.clicked.connect(self.labelUp)
        self.upShortcut = QShortcut("w",self)
        self.upShortcut.activated.connect(self.labelUp)

        self.downButton = QPushButton("Previous membrane (s)")
        self.downButton.clicked.connect(self.labelDown)
        self.downShortcut = QShortcut("s",self)
        self.downShortcut.activated.connect(self.labelDown)

        self.nextFileButton = QPushButton("Next file (d)")
        self.nextFileButton.clicked.connect(lambda: self.openNextImage(1))
        self.nextFileShortcut = QShortcut("d",self)
        self.nextFileShortcut.activated.connect(lambda: self.openNextImage(1))

        self.previousFileButton = QPushButton("Previous file (a)")
        
        self.previousFileButton.clicked.connect(lambda: self.openNextImage(-1))
        self.previousFileShortcut = QShortcut("a",self)
        self.previousFileShortcut.activated.connect(lambda: self.openNextImage(-1))

        self.filesLayout = QHBoxLayout()
        self.filesLayout.addWidget(self.loadFilesButton)
        self.filesLayout.addWidget(self.chooseSaveDirButton)

        self.stackModifierLayout = QHBoxLayout()
        self.stackModifierLayout.addWidget(self.downButton)
        self.stackModifierLayout.addWidget(self.upButton)

        
        

        self.nextPreviousFileLayout = QHBoxLayout()
        self.nextPreviousFileLayout.addWidget(self.previousFileButton)
        self.nextPreviousFileLayout.addWidget(self.nextFileButton)
        


        self.applyLowPassFilterButton = QPushButton("Apply low pass filter")
        self.applyLowPassFilterLineEdit  = QLineEdit("0.1")
        self.applyLowPassFilterLineEdit.setValidator(QDoubleValidator(top=1.0))

        self.applyLowPassFilterButton.clicked.connect(self.applyLowPassFilter)

        self.applyLowPassFilterLayout = QHBoxLayout()
        self.applyLowPassFilterLayout.addWidget(self.applyLowPassFilterButton)
        self.applyLowPassFilterLayout.addWidget(self.applyLowPassFilterLineEdit)


        self.applyHighPassFilterButton = QPushButton("Apply high pass filter")
        # self.applyHighPassFilterLineEdit  = QLineEdit("0.1")
        # self.applyHighPassFilterLineEdit.setValidator(QDoubleValidator(top=1.0))

        self.applyHighPassFilterButton.clicked.connect(self.applyHighPassFilter)

        self.applyHighPassFilterLayout = QHBoxLayout()
        self.applyHighPassFilterLayout.addWidget(self.applyHighPassFilterButton)
        # self.applyHighPassFilterLayout.addWidget(self.applyHighPassFilterLineEdit)


        self.layout().addLayout(self.filesLayout)
        self.layout().addWidget(self.saveDirLabel)
        self.layout().addLayout(self.stackModifierLayout)
        self.layout().addLayout(self.nextPreviousFileLayout)
        self.layout().addLayout(self.applyLowPassFilterLayout)
        self.layout().addLayout(self.applyHighPassFilterLayout)
        if len(self.files) > 0:
            self.openNextImage()
        # self.testButton = QPushButton("test")
        # self.testButton.clicked.connect(self.test)
        # self.layout().addWidget(self.testButton)

    @property
    def saveDir(self):
        return Path(self.saveDir_)

    @saveDir.setter
    def saveDir(self, value):
        if len(str(value)) > 23:
            text = str(value)[:10] + "..." + str(value)[-10:]
        else:
            text = value 
        self.saveDirLabel.setText("Save Directory: " + str(text))
        self.saveDir_ = Path(value)
    
    def loadFiles(self):
        """
        Loads in files given by a QFileDialog and opens the first one.
        Parameters
        ----------


        Returns
        -------
        
        """
        dlg = QFileDialog()

        file_suffixes = " *".join([".mrc", ".rec", ".MRC", ".REC", ".png", ".jpg", ".jpeg"])
        file_suffixes = f"Micrographs (*{file_suffixes})"
        dlg.setFileMode(QFileDialog.ExistingFiles)
        files, filt = dlg.getOpenFileNames(self, "Choose micrographs", ".",filter=file_suffixes)

        if len(files) > 0:
            load = len(self.files) == 0

            files = [Path(file) for file in files]
            if any([file.suffix.lower() not in [".mrc", ".rec"] for file in files]):
                dialog = QDialog()
                dialog.setWindowTitle("None mrc files, choose pixel size.")
                dialog.setLayout(QVBoxLayout())
                label = QLabel("You have chosen files which are not mrc files.\nPlease give a pixel size. (Will be used for all non mrc files)")
                lineedit = QLineEdit(str(self.nonMrcPixelSize))
                lineedit.setValidator(DefaultFloatValidator(self.nonMrcPixelSize))
                button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                button_box.accepted.connect(dialog.accept)
                button_box.rejected.connect(dialog.reject)
                layout = dialog.layout()
                layout.addWidget(label)
                layout.addWidget(lineedit)
                layout.addWidget(button_box)
                if dialog.exec_() == QDialog.Accepted:
                    self.nonMrcPixelSize = float(lineedit.text())
                else:
                    return
            self.files.extend(files)

            if load:
                self.idx = -1
                self.openNextImage()

    def chooseSaveDir(self):
        """
        Opens a file dialog to chose a save directory
        Parameters
        ----------


        Returns
        -------
        
        """
        directory = QFileDialog.getExistingDirectory(caption="Choose directory for saving segmentation files")
        if not isinstance(directory, (str, Path)) or len(directory) == 0:

            return
        directory = Path(directory)
        if not Path().is_dir():
            return
        self.saveDir = directory

    def closeLayers(self):
        """
        Closes the current layers.
        Parameters
        ----------


        Returns
        -------
        
        """
        name = self.getName()
        seg_name = self.getSegmentationName()
        
        
        try:
            del self.viewer.layers[name]

        except Exception as e:
            pass

        try:
            del self.viewer.layers[seg_name]
        except Exception as e:
            pass


    def openNextImage(self, direction=1):
        """
        Opens the next image and saves the current segmentation
        Parameters
        ----------
        direction : int: the direction of which is the next file 

        Returns
        -------
        
        """
        if len(self.files) == 0:
            return
        if isinstance(direction, bool):
            raise ValueError
        
        if not self.saveSegmentation():
            return
        self.closeLayers()
        self.idx += direction
        if self.idx < 0:            
            self.idx = len(self.files) - 1
        elif self.idx >= len(self.files):
            self.idx = 0

        name = self.getName()
        segName = self.getSegmentationName()
        if name is None or segName is None:
            return
        
        # layer = self.viewer.open(self.files[self.idx], plugin="mrcfile-reader")[0]
        # layer = self.viewer.open(self.files[self.idx], plugin=mrc_reader_function)[0]
        if self.files[self.idx].suffix.lower() in [".mrc", ".rec"]:
            read = mrc_reader_function(self.files[self.idx])[0]
            mrc_data = read[0]
            pixel_size = read[1]["metadata"]["pixel_spacing"]
        else:
            mrc_data = load_file(self.files[self.idx])
            pixel_size = self.nonMrcPixelSize

        if not np.isclose(pixel_size, self.pixelSize):
            shape = [int(s * pixel_size / self.pixelSize) for s in mrc_data.shape]
            mrc_data = fft_rescale_image(mrc_data, shape)


        layer = self.viewer.add_image(mrc_data)
        
        layer.name = name
        data = layer.data
        shape = data.shape


        if self.getSegmentationPath() is not None and self.getSegmentationPath().exists():
            seg_data = sparse.load_npz(self.getSegmentationPath()).todense()
        else:
            seg_data = np.zeros((1, *shape), dtype=np.int16)
        label = self.viewer.add_labels(seg_data, name=segName)
        
        
        self.calcStack()

        try:
            name_idx = self.viewer.layers.index(self.getName())
            seg_idx = self.viewer.layers.index(self.getSegmentationName())
            stack_idx = self.viewer.layers.index("SegmentationStack")

            self.viewer.layers.move_multiple(( name_idx,seg_idx, stack_idx ),0)
        except Exception as e:
            print(e)
        self.viewer.layers.selection.active = label
        label.mode = "paint"
        
    def shortenString(self, string, max_length=20):
        """
        Shortens a string to a maximum size.
        Parameters
        ----------
        string     : the string to shorten
        max_length : the maximum size of the string

        Returns
        -------
        string     : the shortened string
        """
        if len(string) <= max_length:
            return string
        else:
            return string[:max_length//2 -1 ] + ".." + string[-max_length//2 +1:]
    
    def getName(self, full=False):
        """
        Returns the name of the current file.
        Parameters
        ----------
        full    : bool: Whether to return the full name or shortened

        Returns
        -------
        name    : the name of the current file
        """
        if len(self.files) > 0:
            file = self.files[self.idx]
            if full:
                return file.stem
            return self.shortenString(file.stem)
        else:
            return None

    def getFileName(self):
        if len(self.files) > 0:
            file = self.files[self.idx]
            return file
        else:
            return None


    def applyLowPassFilter(self):
        """
        Applies low pass filter of the current file.
        Parameters
        ----------


        Returns
        -------
        
        """
        def low_pass_filter(image, cutoff_frequency):
            # Compute the 2D Fourier transform of the image
            image_fft = fft2(image)
            
            # Create a meshgrid of frequency coordinates
            freq_x = np.fft.fftfreq(image.shape[1], 1)
            freq_y = np.fft.fftfreq(image.shape[0], 1)
            freq_meshgrid = np.meshgrid(freq_x, freq_y)
            frequencies = np.sqrt(freq_meshgrid[0]**2 + freq_meshgrid[1]**2)

            
            # Apply the low-pass filter in the Fourier domain
            image_fft_filtered = image_fft * (frequencies <= cutoff_frequency)

            
            # Compute the inverse Fourier transform to obtain the filtered image
            filtered_image = np.real(ifft2(image_fft_filtered))
            
            return filtered_image
        name = self.getName()
        if name is None:
            return
        try:
            layer = self.viewer.layers[name]
        except Exception as e:
            print(e)
            return
        
        # if self.files[self.idx].suffix.lower() in [".mrc", ".rec"]:
        #     read = mrc_reader_function(self.files[self.idx])[0]
        #     img = read[0]
        #     pixel_size = read[1]["metadata"]["pixel_spacing"]
        # else:
        #     img = load_file(self.files[self.idx])
        #     pixel_size = self.nonMrcPixelSize

        # if not np.isclose(pixel_size, self.pixelSize):
        #     shape = [int(s * pixel_size / self.pixelSize) for s in img.shape]
        #     img = fft_rescale_image(img, shape)
        img = layer.data
        low_passed = low_pass_filter(img, float(self.applyLowPassFilterLineEdit.text()))
        layer.data = low_passed

    

    def applyHighPassFilter(self):
        """
        Applies high pass filter of the current file.
        Parameters
        ----------


        Returns
        -------
        
        """
        def gauss(fx,fy,sig):

            r = np.fft.fftshift(np.sqrt(fx**2 + fy**2))
            
            return np.exp(-2*np.pi**2*(r*sig)**2)

        def gaussian_filter(im,sig,apix):
            '''
                sig (real space) and apix in angstrom
            '''
            sig = sig/2/np.pi
            fx,fy = np.meshgrid(np.fft.fftfreq(im.shape[1],apix),\
                                np.fft.fftfreq(im.shape[0],apix))

            im_fft = np.fft.fftshift(np.fft.fft2(im))
            fil = gauss(fx,fy,sig*apix)
            
            im_fft_filtered = im_fft*fil
            newim = np.real(np.fft.ifft2(np.fft.ifftshift(im_fft_filtered)))
            
            return newim
        
        name = self.getName()
        if name is None:
            return
        try:
            layer = self.viewer.layers[name]
        except Exception as e:
            print(e)
            return
        

        img = layer.data
        sig = int(1500 / self.pixelSize)
        sig += (sig + 1) % 2
        high_passed = gaussian_filter(img,0,self.pixelSize) - gaussian_filter(img,sig,self.pixelSize)
        layer.data = high_passed





    def getSegmentationName(self, full=False):
        name = self.getName(full=full)
        if name is None:
            return
        return name + "_labels"

    def getSegmentationPath(self):
        name = self.getSegmentationName(full=True)
        if name is None:
            return
        return self.saveDir / (name + ".npz")

    def saveSegmentation(self):
        """
        Saves the current segmentation.
        Parameters
        ----------


        Returns
        -------
        
        """
        try:
            segmentation = self.viewer.layers[self.getSegmentationName()].data
        except Exception as e:
            pass
            # print(e)
            return True
        
        s = np.sum(segmentation)
        if s == 0 and self.askAgain:
            messageBox = QMessageBox()
            cb = QCheckBox("Do not ask me again this session")
            messageBox.setCheckBox(cb)

            title = "Are you sure?"    
            message = f"There is no segmentation in the current label layer. Do you want to save an empty segmentation?"
            
            messageBox.addButton( messageBox.Yes)
            messageBox.addButton(messageBox.No)
            messageBox.addButton(messageBox.Cancel)
            messageBox.setText(message)
            messageBox.setWindowTitle(title)
            messageBox.setDefaultButton(messageBox.Yes)
            reply = messageBox.exec()
            

            # reply = messageBox.question(self, title, message, messageBox.Yes | messageBox.No | messageBox.Cancel, messageBox.No)
            self.askAgain = not cb.isChecked()
            if not self.askAgain:
                self.answer = reply == messageBox.Yes
            if reply == messageBox.No:
                return True
            if reply == messageBox.Cancel:
                return False

        if s > 0 or self.answer:
            seg_name = self.getSegmentationPath()
            segmentation = self.createThreeDStack(segmentation)
            segmentation = sparse.as_coo(segmentation)

            sparse.save_npz(seg_name, segmentation)
            filename = self.getFileName()
            if filename is not None:
                if self.segmentationModel is not None:
                    self.segmentationModel.updateTrainPaths([filename], [seg_name])
                self.finishedSegmentations[0].append(filename)
                self.finishedSegmentations[1].append(seg_name)
        return True

    def labelUp(self, dummy=None):
        """
        Adds a new layer to the segmentation and increases the label counter by one
        Parameters
        ----------


        Returns
        -------
        
        """
        segname = self.getSegmentationName()

        if len(segname) > 0 and segname in self.viewer.layers:
            
            current_slice = self.viewer.dims.current_step[0] + 1 
            nr_of_slices = self.viewer.layers[segname].data.shape[0]
            if current_slice >= nr_of_slices:
                new_slice = np.zeros((1,*self.viewer.layers[segname].data[0].shape), dtype=bool) 
                self.viewer.layers[segname].data = np.concatenate((self.viewer.layers[segname].data, new_slice))
                self.calcStack()
            old_step = self.viewer.dims.current_step
            self.viewer.dims.current_step = (current_slice, old_step[0], old_step[1])
           
    
    def labelDown(self, dummy=None):
        """
        Goes to the previous label layer of the segmentation.
        Parameters
        ----------


        Returns
        -------
        
        """
        segname = self.getSegmentationName()

        if len(segname) > 0 and segname in self.viewer.layers:
            current_slice = self.viewer.dims.current_step[0] - 1
            if current_slice < 0:
                pass
            else:
                old_step = self.viewer.dims.current_step
                self.viewer.dims.current_step = (current_slice, old_step[0], old_step[1])
            

    def createThreeDStack(self, data):
        """
        Creates a stack of segmentation images. One image for each connected component.
        Parameters
        ----------


        Returns
        -------
        
        """
        def solveLayer(layer):
            unique_labels = np.unique(layer)
            if len(unique_labels) > 2:
                unique_labels = unique_labels[1:]
                new_stack = []
                
                for ul in unique_labels:
                    new_layer = (layer == ul)*1
                    new_stack.append(solveLayer(new_layer))
                return np.concatenate(new_stack)
            labels, num_features = label(layer)
            
            if num_features <= 1 :
                return np.expand_dims(labels,0)
            
            
            new_stack = []
            for label_counter in range(1,num_features + 1):
                label_image = (labels == label_counter) * 1
                # negative_label_image = np.pad(label_image,1) != 1
                # neg_labels, neg_features = label(negative_label_image, np.array([[0,1,0],[1,1,1],[0,1,0]]))
                # neg_labels = neg_labels[1:-1,1:-1]
                # for i in range(neg_features):
                #     if np.sum(neg_labels == (i+1)) <= self.max_hole_size:
                #         label_image[neg_labels == (i+1)] = 1


                new_stack.append(label_image)
            return np.array(new_stack)

        
        if data.ndim == 2:
            
            new_stack = solveLayer(data)
            
        else:
        
            new_stack = []
            for layer in data:
                new_layer = solveLayer(layer)
                if new_layer is not None:
                    new_stack.append(new_layer)
            
            new_stack = np.concatenate(new_stack)
        
        
        return new_stack


    def calcStack(self, dummy=None):
        """
        Calculates the shown segmentation stack as a summed up image.
        Parameters
        ----------


        Returns
        -------
        
        """
        segname = self.getSegmentationName()

        if len(segname) > 0 and segname in self.viewer.layers:
            
                
                if not "SegmentationStack" in self.viewer.layers:
                    current_labels = self.viewer.layers[segname].data
                    summed_image = np.sum(current_labels, axis=0)

                    segstack = self.viewer.add_labels(summed_image, name="SegmentationStack")
                    
                    segstack.selected_label = 5
                    segstack.editable = False
                else:
                    
                    self.viewer.layers["SegmentationStack"].visible = True
                    current_labels = self.viewer.layers[segname].data
                    summed_image = np.sum(current_labels, axis=0)
                    self.viewer.layers["SegmentationStack"].data = summed_image
                    
                if self.viewer.layers[segname] not in self.viewer.layers.selection:
                    self.viewer.layers.selection.active = self.viewer.layers[segname] 
                
                

    def removeSlice(self, dummy):
        """
        Remove a slice of the segmentation stack.
        Parameters
        ----------


        Returns
        -------
        
        """
        if len(self.current_label_layer_name) > 0:
            if self.threed_checkbox.isChecked():
                nr_of_slices = self.viewer.layers[self.current_label_layer_name].data.shape[0]
                if nr_of_slices > 1:

                    current_slice = self.viewer.dims.current_step[0]
                    
                    self.viewer.layers[self.current_label_layer_name].data = np.delete(self.viewer.layers[self.current_label_layer_name].data, current_slice, axis=0)
                    if current_slice > self.viewer.layers[self.current_label_layer_name].data.shape[-1] - 1:
                        old_step = self.viewer.dims.current_step
                        self.viewer.dims.current_step = (self.viewer.layers[self.current_label_layer_name].data.shape[-1] - 1, old_step[0], old_step[1])


    # def closeEvent(self, a0) -> None:
        
    #     if self.custom_parent is not None:
    #         self.custom_parent.closedNapari(self.finishedSegmentations, self.segmentationModel)
    #     return super().closeEvent(a0)