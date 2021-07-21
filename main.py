"""
This program uses three given csv-files (test, training, ideal) that consist of values represented by x-y-pairs.
Each file contains a different amount of datasets. The Training-File (A) contains 4 datasets, the Test-File (B) 1 dataset and the
Ideal-File (C) contains 50 datasets that represent 50 ideal functions.

The program uses training data to choose 4 ideal functions that represent the best fit from all the 50 provided functions in file C.

To choose the ideal functions two criteria on how they are chosen:
------------
    a) how they minimize the sum of all y-deviations squared (Least-Square)

    b) The criterion for mapping the individual test case to the four ideal functions is that the existing maximum deviation
       of the calculated regression does not exceed the largest deviation between training dataset (A) and the ideal function (C)
       chosen for it by more than  factor  sqrt(2)

Furthermore, the program compiles a SQLlite database that loads the training data into a spreadsheet with five columns, where the first column
shows the x values. Correspondingly, the 50 ideal functions are loaded and depicted in the same way.

The test data (B) will afterwards be loaded and matched to one of the four chosen functions, represented by another 4-column SQL-database.


Lastly, the chosen ideal functions as well as the training and test data are visualized via different plotting methods.
"""

import itertools
import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.palettes import Viridis6 as palette #@UnresolvedImport
from bokeh.plotting import figure, output_file, show
from sqlalchemy import create_engine








class Load_Data:

    """
    This class loads the Data Set

    Arguments:

                x_value:        The x value from the dataset will be assigned to this variable
                location:       location from where the dataset is loaded
                data:           the loaded dataset

    """

    def __init__(self, location, x_value="x"):
        self.x_value = x_value
        self.location = location
        self.data = self.load_file(location, x_value)

        """
        This function sets up the attributes for the Data class

        Parameters:
        
                    x_value:        The x value from the dataset will be assigned to this variable
                    location:       location from where the dataset is loaded from
                    data:           the loaded dataset
            
            
        Method:
        
                    load_file:      assigns the datafile to the dataframe
            
        """

    def load_file(self, location,x_value="x"):

        """
        The data is loaded in the pandas dataframe with this method.
        Load the data into a dataframe. Columns with no values are cut off and an error will be returned if there is no
        x value via try and except.

        """

        try:
            x_value
            data = pd.read_csv(location)

            for column in data:
                if data[column].dtype != float: #non-float values are cut off
                    data.dropna(axis=1, how="all")


        except Exception as error:
            print(error)
            quit()
        else:
            return data


class Training(Load_Data):

    """

    This class creates the new object Training data and inherits the properties from the Load_Data class above.

    Attributes:

                dataset:        a merged table of the training dataset and the ideal dataset
                bestfit:        a table of each function and the corresponding ideal function
                results:        The results for a function with another function. Sum of difference squared

    Methods:

                merge_data:     creates an inner join with a second dataset
                least_squares:  the sum of differences squared between the datasets
                fit:            best fit function for every training set
                max_dev:        maximum deviation for every pairing of data sets
                plot_figures:   creates different plots of each function pairing
                create_dbtable: Create a table in the database with the training dataset
                match_ideal:    functions in class to match and fit, plot, and move data to database

    """

    def __init__(self, location, x_value="x"):

        """
        sets up the attributes

        """

        super().__init__(location, x_value) #returns temporary object that references to parent class (Load_data) and
                                            #extends it
        self.dataset = pd.DataFrame()
        self.bestfit = pd.DataFrame()
        self.results = pd.DataFrame()



    def merge_data(self, x_value, idealfunction):

        """
        merges the data with a second dataset on the x functions by creating an inner join

        """
        ideal_x_value = idealfunction.x_value
        ideal_data = idealfunction.data

        dataset = pd.merge(
            self.data,
            ideal_data,
            left_on=x_value, #Column or index level names to join on in the left DataFrame
            right_on=ideal_x_value, #Column or index level names to join on in the left DataFrame
            how="inner", #use intersection of keys from both frames, similar to a SQL inner join
            suffixes=(" (training function)", None)
        )
        return dataset

    def least_squares(self, data, dataset, results):

        """
        Calculates the sum of the difference between values in two datasets and squares them for each pairing
        in the training set and the ideal set.

        """
        for i in range(1, len(data.columns)):  # iterates through every column in the training set with iloc
            training_column = dataset.iloc[:, i] #iloc gets rows (and/or columns) at integer locations
            square_results = []
            for j in range(len(data.columns), len(dataset.columns)):  # iterates through every column in the Ideal dataset, which start after the training columns
                ideal_column = dataset.iloc[:, j]
                squared_sums = ((ideal_column - training_column) ** 2).sum()  # calculates the difference between the values in each dataset,
                                                                      # and squares it. It is later added to the result
                square_results.append([dataset.columns[j], squared_sums])
            results[dataset.columns[i]] = square_results
        return results

    def fit(self, results, bestfit):

        """
        with the calculated results, this finds the best fitting data for every training data

        """

        for val in results:
            for k in range(len(results[val])):
                if results[val][k][1] == min(z[1] for z in results[val]):
                    # search the list of results for the result with the least value
                    bestfit = bestfit.append(
                        pd.DataFrame(
                            [[val, results[val][k][0], results[val][k][1]]],
                            columns=["training func","ideal func","score"]
                        )
                    )
        bestfit.reset_index(drop=True, inplace=True)
        return bestfit

    def max_dev(self, bestfit, dataset):

        """
        Calculates the maximum deviation for matching functions with the training set

        """

        bestfit["max deviations"] = list(pd.DataFrame(dataset[ bestfit["ideal func"]].values - dataset[bestfit["training func"]].values)
                .abs() #returns absolute value
                .max(axis=0) #computes the column maxima
        )
        return bestfit

    def plot_figures(self, bestfit, dataset, x_value):

        """
        This function creates figures of the training function and the matched ideal function.
        The figure represents the training and the ideal function with respect to their x values.

        """

        for match in bestfit.iterrows():  # iterate through all entries in bestfit
            output_file(match[1][0] + ".html") # takes the name of the function to create a filename
            x = dataset[x_value]
            y_train = dataset[match[1][0]]
            y_ideal = dataset[match[1][1]]

            fig1 = figure(
                plot_width=500,
                plot_height=500,
                x_axis_label="X-Values",
                y_axis_label="Y-Values",
                title="Training Function and Ideal Function",
            )
            fig1.scatter(
                x,
                y_train,
                size=2,
                color="blue",
                legend_label=match[1][0],
                marker="circle",
            )
            fig1.scatter(
                x,
                y_ideal,
                size=2,
                color="red",
                legend_label=match[1][1],
                marker="circle",
            )
            fig1.legend.location = "top_left"

            grid = gridplot(children=[[fig1]])
            show(grid)

    def create_dbtable(self, data, connect_sqlite):

        """
        Sets up tables in the database

        """

        sql_table = "Training Data"
        data.to_sql(sql_table, connect_sqlite, if_exists="replace")

    def match_ideal(self, idealfunc, connect_sqlite):

        """
        Runs functions in class to match, fit, plot, and export data
        values of used methods is stored in respective self.xxxx

        """
        self.dataset = self.merge_data(self.x_value, idealfunc)
        self.results = self.least_squares(self.data, self.dataset, self.results)
        self.bestfit = self.fit(self.results, self.bestfit)
        self.bestfit = self.max_dev(self.bestfit, self.dataset)
        self.plot_figures(self.bestfit, self.dataset, self.x_value)
        self.create_dbtable(self.data, connect_sqlite)

        return

class Ideal(Load_Data):

    """
    creates ideal datatable
    inherited attributes from the Load_Data class

    """

    def __init__(self, location, x_value="x"):
        super().__init__(location, x_value)

    def create_dbtable(self, connect_sqlite):

        """
        Create tables in the database with the Ideal data

        """

        sql_table = "Ideal Data"
        self.data.to_sql(sql_table, connect_sqlite, if_exists="replace")


class Test(Load_Data):

    """
    Class to create a Test data object, inheriting from the Data class

    Attributes:

                y_value:                        Column containing the y values
                data["Ideal Func #"]:           New column in the data attribute containing the # corresponding to the matched Ideal function
                data["max_dev Y (test func)"]:  New column in the data attribute containing the difference between the test datapoint



    Methods:

                find_match:                     Merge dataset with the functions from the ideal dataset. For each point in original dataset, find the closest point in one of the given ideal datasets
                plot_figures:                   Create a plot with the original dataset (black) and all the ideal datasets
                create_dbtable:                 Create a table in the database containing the original dataset,
                match_test:                     Runs functions in class to match, fit, plot, and export data

    """

    def __init__(self, location, x_value="x"):

        """
        Constructs all the necessary attributes for the Test object.

        Parameters:

                y_value:                        Column containing the Y values
                data["Ideal Func #"]:           New column in the data attribute containing the # corresponding to the matched Ideal function
                data["max_dev Y (test func)"]:  New column in the data attribute containing the difference between the test datapoint and the matched ideal function datapoint
                functions:                      List containing all the matched functions

        """

        super().__init__(location, x_value)
        self.y_value = self.data.columns.drop(x_value)[0]
        self.data["max_dev Y (test func)"] = np.nan
        self.data["Ideal Func #"] = np.nan
        self.data["Matched Value"] = np.nan
        self.functions = []

    def find_match(
            self,
            data,
            x_value,
            y_value,
            fitting_functions,
            ideal_data,
            ideal_x_value,
            functions,
    ):

        """
        Iterate through the functions that matched in training.
        Left join the data from these functions to the test dataset (left join to keep all datapoints in the test set).
        Search through the Ideal dataset for matching data (difference no bigger than sqrt(2) * max deviation found in training).
        If a value has multiple matches, the new matches are copied to a new row.

        """

        for (item) in (fitting_functions.iterrows()):                  # iteration through the functions in training.bestfit
            row = item[1]
            ideal_func = row[1]                                        # selects the matching ideal function
            data = data.merge(ideal_data[[ideal_x_value, ideal_func]], # merges the function to the data
                left_on=x_value,
                right_on=ideal_x_value,
                how="left",
                suffixes=(" (test func)", None)
            )
            max_deviation = row[3] * np.sqrt(2)                         # max. allowed difference multiplied with sqrt(2)
            diff = data[y_value] - data[ideal_func]                 # calculate the difference of the training y-value with the ideal function
            abs_fit = (diff.abs() <= max_deviation)                     # compares the absolute value difference with the max. allowed difference
            unmatched_values = data["max_dev Y (test func)"].isna()     # checks for unmatched data points
            duplicate_values = abs_fit & np.invert(unmatched_values)    # chooses already matched values
            append_data = data[duplicate_values]                        # copies the old matches
            data.loc[abs_fit, ["max_dev Y (test func)"]] = diff         # insert value, deviation and matched function, overwriting any previously found matches
            data.loc[abs_fit, ["Ideal Func #"]] = ideal_func
            data.loc[abs_fit, ["Matched Value"]] = data[ideal_func]
            data = data.append(append_data)                             # adds older matches again
            data.reset_index(inplace=True, drop=True)

        return data

    def plot_figures(self, data, functions, x_value, y_value):

        """
        Create various plots with the original dataset and the ideal datasets or the matched datapoints.

        Figures:

                figure 3:       A plot of the Test dataset and all 4 ideal datasets
                figure 4:       A plot of the test dataset and the datapoints from the Ideal sets that match

        """

        output_file("matching_testdata.html")
        source = ColumnDataSource(data=data[[x_value, *functions]]) # create a data source for our bokeh plot that includes the x column and all function columns
        colors = itertools.cycle(palette)

        fig3 = figure(
            plot_width=500,
            plot_height=500,
            x_axis_label="X value",
            y_axis_label="Y value",
            title="Test function and all Ideal function matches",
        )
        for y_value, color in zip(functions, colors):
            fig3.scatter(
                source=source,
                x=x_value,
                y=y_value,
                size=4,
                color=color,
                legend_label=y_value,
            )
        fig3.scatter(
            x=data[x_value],
            y=data[y_value],
            size=6,
            color="black",
            legend_label="Test Data",
            marker="triangle",
        )
        fig3.legend.location = "top_center"

        fig4 = figure(
            plot_width=500,
            plot_height=500,
            x_axis_label="X value",
            y_axis_label="Y value",
            title="Test function and matched points",
        )
        fig4.scatter(
            x=data[x_value].loc[
                data["Matched Value"].isna()
            ],  # select only the datapoints from the test function that have no match
            y=data[y_value].loc[data["Matched Value"].isna()],
            size=6,
            color="black",
            legend_label="Unmatched Test Data",
            marker="triangle",
        )
        fig4.scatter(
            x=data[x_value],
            y=data["Matched Value"],
            size=8,
            color="red",
            legend_label="Ideal Data",
            marker="diamond",
        )
        fig4.scatter(
            x=data[x_value].loc[
                data["Matched Value"].isna()
                == False  # selects matching data points
                ],
            y=data[y_value].loc[data["Matched Value"].isna() == False],
            size=4,
            color="blue",
            legend_label="Matched Test Data",
            marker="circle",
        )

        grid = gridplot(children=[[fig3, fig4]])
        show(grid)

    def create_dbtable(self, data, sql_connection):

        """
        Create a table in the database containing the original dataset,
        the matched function for each point and its deviation

        """
        sql_table = "Test Data"
        data.iloc[:, :4].to_sql(sql_table, sql_connection, if_exists="replace")  # write only the relevant data to the table

    def match_test(self, training, ideal, sql_connection):

        """
        Runs functions in class to match and plot and export data to database

        """
        self.data = self.find_match(
            self.data,
            self.x_value,
            self.y_value,
            training.bestfit,
            ideal.data,
            ideal.x_value,
            self.functions,
        )
        self.plot_figures(self.data, self.functions, self.x_value, self.y_value)
        self.create_dbtable(self.data, sql_connection)


def match_database(
        training_loc,
        ideal_loc,
        test_loc,
        training_x="x",
        ideal_x="x",
        test_x="x",
):
    """
    Create a database and use a training, ideal and test dataset to find the best matches and writes values into tables in the database

    parameters:

                training_loc:           location of the training file
                ideal_loc:              location of the ideal file
                test_loc:               location of the test file
                training_x:             x variable in the training set
                ideal_x:                x variable in the ideal set
                test_x:                 x variable in the test set

    """

    engine = create_engine("sqlite:///matching_functions.db", echo=False)
    sql_connect = engine.connect()
    training = Training(training_loc, training_x)
    ideal = Ideal(ideal_loc, ideal_x)
    test = Test(test_loc, test_x)
    training.match_ideal(ideal, sql_connect)
    ideal.create_dbtable(sql_connect)
    test.match_test(training, ideal, sql_connect)

if __name__ == "__main__":  #program doesn't run during matching
    match_database(
        "train.csv",
        "ideal.csv",
        "test.csv",
    )

