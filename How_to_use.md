# Guide on how to use CryoVia

## Installation

See README.md

## Usage

Activate the conda environment:

```conda activate cryovia```

Start CryoVia:
```cryovia```
Optional parameters:
```--njobs 5```
```--gpus 0,1```

You will see the starting menu of CryoVia:

![Starting menu](example_screenshots/starting_menu_a.png "Starting menu")

1. Membrane analyser: The main part of CryoVia. Here you can create Datasets, run analysis and analyse the resulting Data.
2. Neural networks: Here you can create new neural networks, train neural networks or modify them. It also provides methods for manual segmentations.
3. Train classifier: Here you can create, modify and train shape classifiers. It also provides methods to creating new shapes for training.
4. Edge detection: Here you can detect grid holes in micrographs. You can also import CryoVia datasets and remove all membranes found outside of the grid holes.

### Membrane analyser

![Membrane analyser](example_screenshots/data_analysis_with_filter_a.png "Membrane analyser")

1. Add micrographs: Opens a file explorer to select micrograph files to add to the currently selected dataset (in 6.)
2. Load in CSV: Opens a file explorer to select a csv file to load in micrographs and optionally segmentions as well to the currently selected dataset. The path to the micrographs should be in the first column and the options path to the segmentations in the second
3. Remove all micrographs: Removes all micrographs of the currently selected dataset.
4. Zip/Unzip: Zips or unzips the dataset. Reduces the file size and number of files needed for this dataset. Running analysis or modifying this dataset needs this dataset to be unzipped.
5. Run analysis: Will open up a parameter window to run segmentation and various analysis on the membranes found in the micrographs.
6. A list of all available datasets
7. New: Opens up a file explorer to selected where the new dataset will be saved and then creates a new dataset.
8. Copy: Create a copy of the currently selected dataset. All needed data will be copied. Can take some time for large datasets.
9. Remove: Removes the currently selected dataset with all its data. Cannot be reveresed.
10. Inspect: Opens a new window where the micrographs and their corresponding segmentation can be viewed. In this window individual membranes or micrographs can be removed from the dataset. See TODO
11. Data >>: Loads the data into the table view to analyse the data. Multiple different datasets can be loaded in at the same time to compare the results.
12. Filters: Here you can create filters to apply to the current loaded datasets. See image for an example. All numerical values are given in Å or 1/Å.
13. Apply: Applies the filters to the currently seen dataset data.
14. Apply to all tabs: Applies the filters to all loaded in data.
15. Clear all filters: Clears all create dfilters in 12
16. Data table view: All the extracted data from the analysis can be viewed here. Every dataset has its own tab. In each tab each found membrane has its own row and each column is a different extracted attribute. Rows can be removed by selection rows and pressing ```del``` on the keyboard. This will prompt to ask if it should be removed only temporarily or permanently.
17. Export single table: Exports the currently selected tab as a csv file.
18. Export all tables: Exports all currently selected tabs as a csv file. Will add the corresponding dataset as a column. 
19. Manual inspection: This will open a window where each cropped membrane will be shown as well as its segmentation. You can also delete membranes from the dataset using this window.
20. Membranes tab: Shows a graph for the currently selected attribute. If you select "Show all" it will add all dataset tabs to the graphs. 
21. Thickness values: Here you can load in all thickness values from every single pixel point for each dataset. Loading this will take some time.
22. Curvature values: Here you can load in all curvature values from every single pixel point for each dataset. Loading this will take some time.
23. Export: You can save the currently shown graph as a png file. You can also do this by right-clicking the graph.
24. Shows the cropped membrane structure from the currently selected row in the table view. You can right click this image to show the membrane in the complete micrograph, highlighted in red.
25. Shows the cropped membrane segmentation from the currently selected row in the table view.
26. Shows the bilayer thickness values along the membrane contour for the currently selected row in the table view.
27. Shows the curvature values along the membrane contour for the currently selected row in the table view.
28. Message board: CryoVia will print some information here during analysis and loading of data.


