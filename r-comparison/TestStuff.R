library(ggplot2)

molecules <- read.table("../molecule_example_scikitlearn/Data/train_500.csv", sep=",", header=TRUE)
ncol(molecules)

# split out values from labels
moles <- molecules[, 2:1777]
moles.species <- molecules[, 1]

moles.pca <- prcomp(moles,
                 center = FALSE,
                 scale = FALSE)

#print(moles.pca)
ncol(moles.pca$x)
scores <- data.frame(moles.species, moles.pca$x)
qplot(x=PC1, y =PC2, data=scores, colour=moles.species) + theme(legend.position="none")