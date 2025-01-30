#########################################
#    Exponential Random Graph Model     #
#           Example Problem             #
#           Gray's Anatomy              #
#                                       #
#   Data: Garry Weissman, 2011          #
#                                       #
#########################################

library("statnet")

#Simple example using a 3-node network
n<-network.initialize(3, directed=T) #generate an empty 3 node network
n[1,2]<-1  #assign a single link between node 1 and node 2
gplot(n)  #plot the network

e1<-ergm(n~edges)  #conduct an ergm using only the edges term (see lecture slides)
summary(e1)

# Load Data: Requires data sets 'gray-adj.csv' & 'gray-attr.csv'
ga.matrix<-as.matrix(read.table("gray-adj.csv", #the name of the adjacency matrix
                                sep=",", #the spreadsheet uses commas to separate cells
                                header=T, #because there is a header with node ID
                                row.names=1, #because the first column has node ID
                                quote="\""))

ga.attr<-read.csv("gray-attr.csv",  #the name of the attributes file
                  header=TRUE, 
                  sep=",", 
                  stringsAsFactors = FALSE)

#Convert the data into a network object in statnet
ga.net<-network(ga.matrix, 
                vertex.attr = ga.attr,
                vertex.attrnames = colnames(ga.attr),
                directed=F, 
                loops=F, 
                multiple=F, 
                bipartite=F, 
                hyper=F)

# Plot the network
plot(ga.net, 
     vertex.col=c("blue","pink")[1+(get.vertex.attribute(ga.net, "sex")=="F")],
     label=get.vertex.attribute(ga.net, "vertex.names"), 
     label.cex=.7)



# PART B: Betweenness Scores and Degree Scores

# Calculate betweenness centrality
betweenness_centrality <- betweenness(ga.net)

# Calculate degree centrality
degree_centrality <- degree(ga.net)

# Find the top 5 nodes for each centrality measure
top_5_betweenness <- sort(betweenness_centrality, decreasing=TRUE)[1:5]
top_5_degree <- sort(degree_centrality, decreasing=TRUE)[1:5]

# Print the results
top_5_betweenness
top_5_degree

# Plot betweenness vs degree
plot(degree_centrality, betweenness_centrality, 
     xlab="Degree Centrality", 
     ylab="Betweenness Centrality", 
     main="Centrality Plot")

# Get the full list of node names
node_names <- network.vertex.names(ga.net)

# Map the top centrality scores back to their respective node names
top_5_betweenness_names <- node_names[order(betweenness_centrality, decreasing=TRUE)[1:5]]
top_5_degree_names <- node_names[order(degree_centrality, decreasing=TRUE)[1:5]]

# Plot the network
plot(ga.net, 
     vertex.cex=sqrt(degree_centrality),   # Size nodes by degree centrality
     vertex.col=c("blue","pink")[1+(get.vertex.attribute(ga.net, "sex")=="F")], # Color nodes by gender
     label=ifelse(network.vertex.names(ga.net) %in% top_5_degree_names, 
                  network.vertex.names(ga.net), ""),  # Display labels only for top 5 Degree Centrality nodes 
     label.cex=.7,
     main="Network Visualization (Top 5 Degree Scores Nodes Labeled)")

# Add a legend
legend("topright", legend=c("Male", "Female"), col=c("blue", "pink"), pch=19)



# Conduct Exponential Random Graph Models (ERGM) Analysis

e2<-ergm(ga.net~edges)  #Create a restricted model with just edges term
summary(e2)

e3<-ergm(ga.net~edges+triangle)  #include a triadic effect in the model
summary(e3)

e4<-ergm(ga.net~edges+triangle+nodematch("sex"))  #Create an unrestricted model
summary(e4)



#PART C: 1
e5<-ergm(ga.net~edges+nodematch("race"))  # Testing for racial homophily
summary(e5)

#PART C: 2
e6 <- ergm(ga.net ~ edges + nodematch("sex")) # Testing for gender homophily
summary(e6)

# Not covered: goodness of fit testing and dealing with degenerate models
