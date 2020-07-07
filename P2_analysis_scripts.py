# Importing librairies

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------------------------------------------------------------------
def world_stats(df, col = 'last_value'):
    """ Returns a table giving standard statistics of the indicator, for each region and for the entire set of countries """
    
    descr_world = df[df['Region'].isna() == False][[col]].describe().round(2).rename(columns = {col : 'World'})
    descr_region = df[df['Region'].isna() == False].pivot(index = 'Country Name', columns='Region',values=col).describe().round(2)

    return descr_world.join(descr_region).T

# -----------------------------------------------------------------------------------------------------------------------------------------
def region_boxplot(df, col = 'last_value'):
    """ Returns seven boxplots (one per region) showing distribution of indicator's last value for each world zone """
    
    # Creation of a dedicated pivot table
    pivot = df[df['Region'].isna() == False].pivot(index = 'Country Name', columns='Region',values= col)
    
    # Plotting 
    region_bp = pivot.boxplot(
        figsize = (15,5), 
        rot = 0
    )
    
    plt.tight_layout()
    # plt.title(df['Indicator Name'].unique()[0])
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------
def world_topN(df, col = 'last_value', world_n = 10, last = True):
    """ Returns a dataframe with top N countries worldwide on given indicator """
    
    # Creating the top 10 table
    top_list = df[df['Region'].isna() == False][[
        'Country Name', col]].sort_values(by = col, ascending = False).head(world_n)['Country Name'].values
    top_score = df[df['Region'].isna() == False][['Country Name', col
                                                 ]].sort_values(by = col, ascending = False).head(world_n)[col].values
    # Addind last 10 if required
    if last == True:
        last_list = df[df['Region'].isna() == False][[
            'Country Name', col]].sort_values(by = col, ascending = True).head(world_n)['Country Name'].values
        last_score = df[df['Region'].isna() == False][[
            'Country Name', col]].sort_values(by = col, ascending = True).head(world_n)[col].values
        
        world_top = pd.DataFrame({'Top countries' : top_list, 'Top scores' : top_score,
                                  'Last countries' : last_list , 'Last scores' : last_score},
                                 index = np.arange(1,world_n + 1,1)) 
    
    else:
        world_top = pd.DataFrame({ 'Top countries' : top_list, 'Top scores' : top_score} , index = np.arange(1,world_n + 1,1))
    
    return world_top
                 
# ----------------------------------------------------------------------------------------------------------------------------------------- 
def map_indic(df, indic_column, color_scale, reverse_scale = True, map_title = "", scale_title = ""):
    """ Creates a map of a given indicator """
    
    fig = go.Figure(data=go.Choropleth(
        
        # Informations
        locations = df[df['Region'].isna() == False]['Country Code'],
        z = df[df['Region'].isna() == False][indic_column],
        text = df[df['Region'].isna() == False]['Country Name'],
        
        # Map design
        colorscale = color_scale,
        reversescale = reverse_scale,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        
        colorbar_title = scale_title,
    ))
    
    fig.update_layout(
        title_text = map_title
    )
    
    fig.show()
    
# ----------------------------------------------------------------------------------------------------------------------------------------- 
def normalize_serie(serie):
    """ Normalize values of a column in df between 0 and 1 """
    
    s_min = serie.min()
    s_max = serie.max()
    
    norm_serie = serie.apply(lambda x : (x-s_min) / (s_max - s_min))
    
    return norm_serie