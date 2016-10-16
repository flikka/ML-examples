


well_outcome <- read.table("well-example.csv", sep=";",row.name = "Well.ID", header=TRUE)



































library(rpart)

fit <- rpart(Success ~ Feature.1 + Feature.2 + Feature.3, data=well_outcome, control=rpart.control(minbucket=1), method="class")

plot(fit)
text(fit, cex=.75)
