import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import re
import plotly.graph_objects as go

# -----------------------------------------------------------------------------------------------------------------------------------------
def calc_sizes(df_list):
    """ Function returning number of rows and columns, and the number of elements for each dataset in the list given as argument
    Also returns a plot representing the number of elements in the datasets """
    
    datasets = []
    dataset_sizes = []

    # Calculating number of rows, columns and elements
    for df in df_list:
        count_rows = df.shape[0]
        count_columns = df.shape[1]
        count_elements = df.size

        datasets.append(df.name)
        dataset_sizes.append(df.size)

        print("The dataset {} contains {} elements, in {} rows and {} columns."
              .format(df.name, count_elements, count_rows, count_columns))
              
    # Visualization of dataset sizes
    hist_size = plt.bar(datasets,dataset_sizes)
    plt.yscale('log')
    plt.title("Size of the datasets")
    plt.ylabel("Number of elements")
    plt.gca().yaxis.grid(True)
    plt.show()
    
# -----------------------------------------------------------------------------------------------------------------------------------------
def calc_missing(df_list):
    """ Function returning the number of NaN elements for each dataset in the list given as argument,
    and the percentage of empty cells in those datasets
    
    Also returns a visualisation showing the number of elements, 
    percentage of empty cells and availiable data for each data set """
    
    # Initialization
    missing_data = []
    availiable_data = []
    dataset_percent_missing = []
    datasets = []
    dataset_sizes = []

    # Calculating availiable and missing data
    for df in df_list:
        count_NaN = df.isnull().sum().sum()
        percent_NaN = 100 * round(count_NaN / df.size,3)
    
        missing_data.append(count_NaN)
        availiable_data.append(df.size - count_NaN)
        dataset_percent_missing.append(percent_NaN)
        datasets.append(df.name)
        dataset_sizes.append(df.size)
        
        print("{} cells in the dataset {} are filled with null elements, representing {}% of the dataset."
          .format(count_NaN, df.name, percent_NaN))
    
    # Visual representation
    fig = plt.figure(1, constrained_layout = True, figsize = (10,7))
    gs = fig.add_gridspec(2, 2)
    fig_ax1 = fig.add_subplot(gs[0, :-1])
    fig_ax2 = fig.add_subplot(gs[1, :-1])
    fig_ax3 = fig.add_subplot(gs[0:, 1:])

    # First plot shows the total amount of elements in datasets
    plt.subplot(fig_ax1, width = 1)
    plt.bar(datasets,dataset_sizes)
    plt.yscale('log')
    plt.title("Size of the datasets")
    plt.ylabel("Number of elements")
    plt.gca().yaxis.grid(True)

    # Second plot gives the percentage of NaN elements in the datasets
    plt.subplot(fig_ax2)
    plt.bar(datasets, dataset_percent_missing)
    plt.title("Percentage of empty data in the datasets")

    # Last plot returns the total amount of actual data in each data set
    plt.subplot(fig_ax3)
    plt.bar(datasets, availiable_data, color = 'green')
    plt.yscale('log')
    plt.title("Availiable data in the datasets")
    plt.gca().yaxis.grid(True)
    
# -----------------------------------------------------------------------------------------------------------------------------------------
def calc_missing2(df_list):
    """ Function returning the number of NaN elements for each dataset in the list given as argument, 
    and the percentage of empty cells in those datasets
    
    Also returns a visualisation showing the number of elements, 
    percentage of empty cells and availiable data for each data set """
    
    # Initialization
    missing_data = []
    availiable_data = []
    dataset_percent_missing = []
    datasets = []
    dataset_sizes = []

    # Calculating availiable and missing data
    for df in df_list:
        count_NaN = df.isnull().sum().sum()
        percent_NaN = 100 * round(count_NaN / df.size,3)
    
        missing_data.append(count_NaN)
        availiable_data.append(df.size - count_NaN)
        dataset_percent_missing.append(percent_NaN)
        datasets.append(df.name)
        dataset_sizes.append(df.size)
        
        print("{} cells in the dataset {} are filled with null elements, representing {}% of the dataset."
          .format(count_NaN, df.name, percent_NaN))
        
    # Visual representation
    pivot = pd.DataFrame({'Dataset sizes':dataset_sizes,
                          'Availiable data':availiable_data},
                        index = datasets)
    pivot.plot.bar(figsize = (10,7), rot = 0)
    plt.yscale('log')
    plt.title("Elements in datasets")
    plt.ylabel("Number of elements")
    plt.gca().yaxis.grid(True)
    
    plt.show()
    
    # Second plot gives the percentage of NaN elements in the datasets
    plt.subplot()
    plt.bar(datasets, dataset_percent_missing)
    plt.title("Percentage of empty data in the datasets")

    plt.show()
    
    # Representing availiable and empty cells in a pie chart for each dataset
    for index in [0,1,2,3,4]:
        plt.pie([missing_data[index], availiable_data[index]], 
            labels = ['null elements', 'availiable data'], 
            colors = ['red','green'], labeldistance = None,
            autopct = lambda x: str(round(x, 2)) + '%', pctdistance = 0.5,
            startangle = 90)
        pie_title = "Availiable data in " + datasets[index]
        plt.title(pie_title)
        plt.legend(loc = 'best')
        plt.show()
    
# -----------------------------------------------------------------------------------------------------------------------------------------
def most_common_words(labels, nb = 20):
    """ Returns the 20 most common words in a DataSeries """
    
    words = []
    for lab in labels:
        words += lab.split(" ")
    counter = Counter(words)
    for word in counter.most_common(nb):
        print(word)
        
# -----------------------------------------------------------------------------------------------------------------------------------------
def check_regex(regex, serie):
    """ Function returning 10 entries of df matching a given regular expression on a given serie """
    
    def check_re(string):
        if re.search(regex,string) is not None:
            check_result = string
        else:
            check_result = False
        return check_result
    
    df_test_result = pd.DataFrame(serie.apply(check_re))
    return df_test_result[df_test_result.iloc[:, 0] != False].head(10)

# -----------------------------------------------------------------------------------------------------------------------------------------
def count_measures(df, year):
    """ for each row of df, count the number of measures since specified year """
    
    counting = df.loc[:,str(year):'2017'].count(axis = 'columns')
    counting.value_counts(sort = False, normalize = True).plot(kind='bar').yaxis.grid(True)
    
    plot_title = "Distribution of number of measures since " + str(year)
    plt.title(plot_title)
    plt.xlabel("x = Number of measures on per row")
    plt.ylabel("Percentage of rows with x measures")

# ----------------------------------------------------------------------------------------------------------------------------------------- 
def drop_years(df, start_year, end_year):
    """ Deletes columns in the df dataset corresponding to years from start_year to end_year (both included in columns to drop).
    Also drop entries with no measures in the time period """
    
    df_wip = df.copy()
    year = start_year
    
    while year <= end_year:
        del df_wip[str(year)]
        year += 1
    
    df_wip['nb_measure'] = df_wip.count(axis = 'columns')-5
    
    return df_wip[df_wip['nb_measure']>0]
    
# ----------------------------------------------------------------------------------------------------------------------------------------- 
def last_measure(df):
    """ Adds two columns to the dataset :
    last_value gives the indicator value of the most recent measure
    last_year returns the year when the last value was measured """
    
    df_wip = df.copy()
    year = int(df_wip.columns[4])

    df_wip['last_year'] = df_wip.loc[:,'2002':'2017'].apply(pd.Series.last_valid_index, axis = 1)

    def last_value(x):
        index= x['last_year']
        if index is not None:
            result = x[index]
        else:
            result = None
        return result

    df_wip['last_value'] = df_wip.apply(last_value, axis = 1)
    df_wip['last_year'] = df_wip['last_year'].apply(lambda x : int(x))
    
    
    return df_wip

# ----------------------------------------------------------------------------------------------------------------------------------------- 
def drop_topics(df, topic_list):
    """ Removes rows in df in which indicator is related to a topic present in topic_list """
    
    df_wip = df.copy()
    initial_size = df_wip.shape[0]
    
    for topic in topic_list:
        df_wip = df_wip[df_wip['Topic'] != topic]
    
    final_size = df_wip.shape[0]
    
    sizes = [initial_size, final_size]
    labels = ['Initial','After drops']
    
    plt.bar(labels, sizes)
    plt.title('Comparing number of rows in df_Data after dropping irrelevant topics')
    plt.ylabel('Number of rows')
    
    plt.rcParams['axes.axisbelow'] = True
    plt.gca().yaxis.grid(True)
    plt.show()
    
    return df_wip
    
# ----------------------------------------------------------------------------------------------------------------------------------------- 
def plot_indicator_distrib(df, indicator):
    """ For the given indicator in df, returns for each year the number of rows in which last_year is the given year """
    
    indic_code = df[df['Indicator Name'] == indicator]['Indicator Code'].unique()[0]
    mean_value = str(round(df[df['Indicator Name'] == indicator]['last_year'].mean()))
    with_indicator = len(df[df['Indicator Name'] == indicator]['Country Code'].unique())
    no_indicator = len(df['Country Code'].unique()) - with_indicator
    mean_measures = str(round(df[df['Indicator Name'] == indicator]['nb_measure'].mean(),1))
    
    print('Indicator : ' + indicator)
    print("Indicator code : " + indic_code)
    
    # Visual representation
    
    plt_wip = df[df['Indicator Name'] == indicator]['last_year'].value_counts(sort=False).plot.bar()
    
    plot_title = "Distribution of last_year for the " + indic_code + " indicator"
    plt.title(plot_title)
    plt.text(0,70, "mean last_year = \n" + mean_value )
    plt.gca().yaxis.grid(True)
    
    plt.show()
    
    # Part of represented countries
    
    plt.pie([no_indicator, with_indicator], 
            labels = [indic_code + ' is not availiable', indic_code + ' is availiable'], 
            colors = ['red','green'], labeldistance = None,
            autopct = lambda x: str(round(x, 2)) + '%', pctdistance = 0.5,
            startangle = 90)
    
    pie_title = "Percentage of countries with availiable " + indic_code + " indicator"
    plt.title(pie_title)
    plt.legend(loc = 'best')
    
    plt.show()
    
    # Distribution of number of measures in the 2002 - 2017 period
    
    plt_measure = df[df['Indicator Name'] == indicator][['Country Code','nb_measure']].groupby(['nb_measure']).count().plot.bar()
    
    plt.title("Distribution of the number of measures between 2002 and 2017 for the \n" +  indic_code + " indicator")
    plt.text(0,40, "mean number of measures = \n" + mean_measures )
    plt.gca().yaxis.grid(True)
    plt.xlabel("x = Number of measures since 2002")
    plt.ylabel("Number of countries with x measures")
    
    plt.show()
    
# ----------------------------------------------------------------------------------------------------------------------------------------- 
def plot_projections(df, indicator):
    """ For the given indicator in df, returns for each year the number of rows in which last_year is the given year """
    
    indic_code = df[df['Indicator Name'] == indicator]['Indicator Code'].unique()[0]
    with_indicator = len(df[df['Indicator Name'] == indicator]['Country Code'].unique())
    no_indicator = len(df['Country Code'].unique()) - with_indicator
    mean_measures = str(round(df[df['Indicator Name'] == indicator]['nb_projected'].mean(),1))
    
    # Part of represented countries
    
    plt.pie([no_indicator, with_indicator], 
            labels = [indic_code + ' is not availiable', indic_code + ' is availiable'], 
            colors = ['red','green'], labeldistance = None,
            autopct = lambda x: str(round(x, 2)) + '%', pctdistance = 0.5,
            startangle = 90)
    
    pie_title = "Percentage of countries with availiable " + indic_code + " indicator"
    plt.title(pie_title)
    plt.legend(loc = 'best')
    
    plt.show()
    
    # Distribution of number of measures from 2020 to 2100
    
    plt_measure = df[df['Indicator Name'] == indicator][['Country Code','nb_projected']].groupby(['nb_projected']).count().plot.bar()
    plt.title("Distribution of the number of projections from 2020 to 2100 for the \n" +  indic_code + " indicator")
    plt.text(0,40, "mean number of projections = \n" + mean_measures )
    plt.gca().yaxis.grid(True)
    plt.xlabel("x = Number of measures from 2100")
    plt.ylabel("Number of countries with x measures")
    
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------- 
def region_mapping(df):
    """ Returns a map of the world with a distinct color for each geographycal region """
    
    # Construction of a dedicated dataset
    df_wip = df[['Country Code', 'Region']].copy()
    df_wip['Region_index'] = df_wip['Region'].str.len()
    
    # Visual representation
    fig = go.Figure(data=go.Choropleth(
        locations = df_wip[df_wip['Region'].isna() == False]['Country Code'],
        z = df_wip[df_wip['Region'].isna() == False]['Region_index'],
        colorscale = 'Rainbow',
        reversescale = False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
    ))
    fig.update_layout(
        title_text = 'World Regions'
    )
    fig.show()