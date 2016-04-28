library(ggplot2)
install_github("ggbiplot", "vqv")

molecules <- read.table("../molecule_example_scikitlearn/Data/train_500.csv", sep=",", header=TRUE)
ncol(molecules)

# split out values from labels
moles <- log(molecules[, 2:1777])
moles.species <- molecules[, 1]

moles.pca <- prcomp(moles,
                 center = FALSE,
                 scale. = FALSE)

plot(moles.pca, type = "l")

qplot(x=PC1, y =PC2, data=moles.pca$x[,1:3], colour=factor(moles.species)) + theme(legend.position="none")