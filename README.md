# IU_Programming_with_Python
Assignment for the Python course

This is the project for the "Programming with Python" course for IU - International University of Applied Science. 

The following text describes the tasks the program needs to execute. The text is quoted from the scribt.

"You are given four training datasets in the form of csv-files.  
YourPython program needs to be able to independently com-pile a SQLite database (file) ideally via sqlalchemy
and load the training data into a single five-column spreadsheet/ table in the file. Its first column depicts the x-values of all functions.
Table 1,at the end of this subsection,shows you which struc-ture your table is expected to have.  
The fifty ideal functions, which are also provided via a CSV-file, must be loaded into another table. 
Likewise, thefirst column depicts the x-values,meaningthere will be 51 columns overall. 
Table 2,at end of this subsection,schematically describes what structure is expected.
After the training data and the ideal functions have been loaded into the database, 
the test data (B) must be loaded line-by-line from another CSV-file and –if it complies with the compiling criterion 
–matched to one of the four functions chosen under i (subsection above). 
Afterwards, the results need to be saved into another four-column-table in the SQLite database.
In accordancewith table 3 at end of this subsection, this table contains four columns with x-and y-values
as well as the cor-responding chosen ideal function and the related deviation.
Finally, the training data, the test data, the chosen ideal functions as well as the corresponding/ assigned datasets
are visualized under an appropriately chosen representation of the deviation.  

Please create a Python-program which also fulfills the following criteria: 
Its design is sensibly object-oriented It includes at least oneinheritance 
It includes standard-und user-defined exception handlingsFor logical reasons
it makes use of Pandas’ packages as well as datavisualization via Bokeh, sqlalchemy,as well as others
Write unit-tests for all useful elements
Your code needs to be documented in its entirety and also include Documentation Strings, known as ”docstrings“

AdditionalTask
Assume that your successfully created project is on the Version Control System Git and has a Branch called develop.
On this Branch, all operations of the developer team are combined. 
Write the Git-commands necessary to clone thebranchand?develop on your local PC. 
Imagine that you have added a new function. 
Write all necessary Git-commands to introduce this project to the team’s develop Branch. 
Please note: You need the commands for commit, push. Afterwards, you would make a Pull-request 
and your contribution would be added “merged” to the developBranch after one or several of your teamhasreviewed your changes."
