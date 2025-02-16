#### ==========================================================================
#### Dissertation Launch D
#### Author: Simon Schramm
#### 13.06.2024
#### --------------------------------------------------------------------------
""" 
This script contains methods to launch the Graph DB "NEO4J".
A prior installation is required: https://neo4j.com/docs/operations-manual/current/installation/
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
from py2neo import Graph, Node, Relationship 
from google.cloud import bigquery
### ---------------------------------------------------------------------------
#%% Methods
### ---------------------------------------------------------------------------
def construct_onedirectional_ekg(first_sale_date, last_sale_date):
    # Connect to Neo4j
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    # Clear existing data
    graph.delete_all()
    #TODO: LOAD DATA DYNAMICALLY.
    # Create nodes
    bmw = Node("Actor", name="BMW")
    tesla = Node("Actor", name="TESLA")
    government = Node("Actor", name="Government")
    #
    bmw_i3 = Node("Vehicle", name="i3", start_of_sales=first_sale_date.date(), end_of_sales=last_sale_date.date())
    tesla_model_3 = Node("Vehicle", name="Model 3", start_of_sales="2019-02-01", end_of_sales="2024-12-31")  # Example dates
    germany = Node("Country", name="Germany")
    subsidy = Node("Subsidy", name="Subsidy")
    # Create relationships
    bmw_i3_sales = Relationship(bmw_i3, "hasSales", germany)
    tesla_model_3_sales = Relationship(tesla_model_3, "hasSales", germany)
    bmw_i3_impact = Relationship(subsidy, "hasImpact", bmw_i3)
    tesla_model_3_impact = Relationship(subsidy, "hasImpact", tesla_model_3)
    bmw_vehicle = Relationship(bmw_i3, "isModelof", bmw)
    tesla_vehicle = Relationship(tesla_model_3, "isModelof", tesla)
    subsidy_actor = Relationship(government, "provides", subsidy)
    # Add properties to the relationships
    bmw_i3_sales['2015-10-10'] = 201
    tesla_model_3_sales['2015-10-10'] = 0
    #
    bmw_i3_sales['2019-04-01'] = 770 
    tesla_model_3_sales['2019-04-01'] = 7515
    #
    bmw_i3_sales['2022-10-08'] = 1175  
    tesla_model_3_sales['2022-10-08'] = 373
    #
    bmw_i3_impact['2015-10-10'] = 4000
    bmw_i3_impact['2019-04-01'] = 6000
    bmw_i3_impact['2022-10-08'] = 4500 
    #
    tesla_model_3_impact['2015-10-10'] = 4000
    tesla_model_3_impact['2019-04-01'] = 6000
    tesla_model_3_impact['2022-10-08'] = 4500 
    #
    # Add nodes and relationships to the graph
    graph.create(bmw | tesla | bmw_i3 | tesla_model_3 | subsidy)
    graph.create(bmw_vehicle | tesla_vehicle | bmw_i3_sales | tesla_model_3_sales | bmw_i3_impact | tesla_model_3_impact)
    return graph
#%%
def construct_temporal_knowledge_graph(df_monthly, list_econ_cols):
    # Connect to Neo4j.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    # Clear existing data.
    graph.delete_all()
    # Create initial nodes.
    bmw = Node("Actor", name="BMW")
    tesla = Node("Actor", name="TESLA")
    # Extract unique countries from the column headers of df_monthly.
    countries = set(col.split(' in ')[-1].strip() for col in df_monthly.columns if 'Sales' in col or 'in Germany' in col)
    # Create country nodes.
    country_nodes = {country: Node("Country", name=country) for country in countries}
    # Extract vehicle names and body styles from the column headers of df_monthly.
    vehicle_columns = [col for col in df_monthly.columns if 'Sales' in col]
    # Create vehicle nodes, body style nodes, and relationships.
    vehicle_nodes = []
    vehicle_relationships = []
    body_style_nodes = {}
    for vehicle in vehicle_columns:
        brand = "BMW" if "BMW" in vehicle else "TESLA"
        country = vehicle.split(' in ')[-1]
        model_name = vehicle.split(' ')[1]  # Extract the model name
        body_style = next((style for style in ['Sedan', 'Convertible', 'Coupe', 'Estate'] if style in vehicle), 'Unknown')
        if body_style not in body_style_nodes:
            body_style_nodes[body_style] = Node("BodyStyle", name=body_style)
        start_of_sales = df_monthly[vehicle][df_monthly[vehicle] > 0].dropna().index.min().date()
        end_of_sales = df_monthly[vehicle][df_monthly[vehicle] > 0].dropna().index.max().date()
        vehicle_node = Node("Vehicle", name=model_name, start_of_sales=start_of_sales, end_of_sales=end_of_sales)
        vehicle_nodes.append(vehicle_node)
        vehicle_relationship = Relationship(vehicle_node, "isModelof", bmw if brand == "BMW" else tesla)
        vehicle_relationships.append(vehicle_relationship)
        body_style_relationship = Relationship(vehicle_node, "hasBodyStyle", body_style_nodes[body_style])
        vehicle_relationships.append(body_style_relationship)
        country_relationship = Relationship(vehicle_node, "hasSales", country_nodes[country])
        sales_data = df_monthly[vehicle].dropna()
        for date, sales in sales_data.items():
            country_relationship[date.strftime('%Y-%m-%d')] = sales
        vehicle_relationships.append(country_relationship)
    # Add economic indicators.
    econ_columns = [col for col in df_monthly.columns if any(econ in col for econ in list_econ_cols)]
    econ_nodes = []
    econ_relationships = []
    for econ in econ_columns:
        country = econ.split(' in ')[-1].strip()
        indicator_name = econ.split(' in ')[0].strip()
        econ_node = Node("EconomicIndicator", name=indicator_name)
        econ_nodes.append(econ_node)
        country_relationship = Relationship(econ_node, "belongsTo", country_nodes[country])
        econ_data = df_monthly[econ].dropna()
        for date, value in econ_data.items():
            country_relationship[date.strftime('%Y-%m-%d')] = value
        econ_relationships.append(country_relationship)
    # Add indexed economic indicators.
    indexed_econ_columns = [col for col in df_monthly.columns if '_indexed' in col]
    indexed_econ_nodes = []
    indexed_econ_relationships = []
    for econ in indexed_econ_columns:
        country = "Germany"
        indicator_name = econ.split('_indexed')[0].strip()
        indexed_econ_node = Node("IndexedEconomicIndicator", name=indicator_name)
        indexed_econ_nodes.append(indexed_econ_node)
        country_relationship = Relationship(indexed_econ_node, "hasImpact", country_nodes[country])
        econ_data = df_monthly[econ].dropna()
        for date, value in econ_data.items():
            country_relationship[date.strftime('%Y-%m-%d')] = value
        indexed_econ_relationships.append(country_relationship)
    # Add nodes and relationships to the graph.
    graph.create(bmw)
    graph.create(tesla)
    for country_node in country_nodes.values():
        graph.create(country_node)
    for body_style_node in body_style_nodes.values():
        graph.create(body_style_node)
    for vehicle_node in vehicle_nodes:
        graph.create(vehicle_node)
    for vehicle_relationship in vehicle_relationships:
        graph.create(vehicle_relationship)
    for econ_node in econ_nodes:
        graph.create(econ_node)
    for econ_relationship in econ_relationships:
        graph.create(econ_relationship)
    for indexed_econ_node in indexed_econ_nodes:
        graph.create(indexed_econ_node)
    for indexed_econ_relationship in indexed_econ_relationships:
        graph.create(indexed_econ_relationship)
    return graph
#
def add_events_to_tkg(df_events, graph):
    # Iterate over each row in the events dataframe
    for index, row in df_events.iterrows():
        # Create a node for each event
        event_node = Node(row['class'], name=row['name'], start=row['start'], end=row['end'])
        graph.create(event_node)
        # Create a relationship to the country node
        country_node = graph.nodes.match("Country", name="Germany").first()
        if country_node:
            relation = Relationship(event_node, "IMPACT_TIV", country_node)
            graph.create(relation)
        # Create relationships to all nodes in detected_in
        detected_in_nodes = row['detected_in'].split(',')
        for detected_node_name in detected_in_nodes:
            detected_node = graph.nodes.match(name=detected_node_name.strip()).first()
            if detected_node:
                relation_tiv = Relationship(event_node, "IMPACT_TIV", detected_node)
                relation_bmw = Relationship(event_node, "IMPACT_BMW", detected_node)
                relation_bev = Relationship(event_node, "IMPACT_BEV", detected_node)
                graph.create(relation_tiv)
                graph.create(relation_bmw)
                graph.create(relation_bev)
    return graph
#
def search_gdelt_for_umweltbonus_events():
    # Initialize a BigQuery client.
    client = bigquery.Client()
    # Define the query to search for events related to "Umweltbonus" in the GDELT KG.
    query = """
    SELECT
        *
    FROM
        `gdelt-bq.gdeltv2.gkg`
    WHERE
        LOWER(V2THEMES) LIKE '%umweltbonus%'
        OR LOWER(V2ENHANCEDTHEMES) LIKE '%umweltbonus%'
        OR LOWER(V2LOCATIONS) LIKE '%germany%'
    LIMIT 1000
    """
    # Execute the query.
    query_job = client.query(query)
    # Fetch the results.
    results = query_job.result()
    # Process and return the results.
    events = []
    for row in results:
        events.append({
            "GKGRECORDID": row.GKGRECORDID,
            "DATE": row.DATE,
            "SourceCollectionIdentifier": row.SourceCollectionIdentifier,
            "SourceCommonName": row.SourceCommonName,
            "DocumentIdentifier": row.DocumentIdentifier,
            "V2Themes": row.V2THEMES,
            "V2EnhancedThemes": row.V2ENHANCEDTHEMES,
            "V2Locations": row.V2LOCATIONS,
            "V2Persons": row.V2Persons,
            "V2Organizations": row.V2Organizations,
            "V2Tone": row.V2Tone
        })
    return events
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================