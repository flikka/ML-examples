library(rpart)

well_outcome <- read.table("well-example.csv", sep=";",row.name = "Well.ID", header=TRUE)

fit <- rpart(Success ~ Feature.1 + Feature.2 + Feature.3, data=well_outcome, control=rpart.control(minbucket=0), method="class")

plot(fit)
text(fit, cex=.75)
