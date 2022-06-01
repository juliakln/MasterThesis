# Clear environment
rm(list = ls())
gc()
# Set working directory to current file location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# load packages
library(ggplot2)
library(RColorBrewer)
library(gridExtra)
library(gtable)
library(grid)
col1 <- brewer.pal(n = 9, name = "Blues")


### 1. experiment
dat <- read.table("hist1_PO.txt", sep=",", fill=TRUE)
scale <- 60

# colony sizes
colony <- dat[1,]
colony <- colony[!is.na(colony)]

# number of stinging bees
stinging <- round(dat[2:nrow(dat),]*scale)

# create and save histograms for each colony size
p <- list()
for(col in 1:length(colony)){
  hist <- data.frame(bees = as.factor(c(0:colony[col])),
                     values = as.integer(stinging[col,!is.na(stinging[col,])]))
  p[[col]] <- ggplot(hist, aes(x = bees, y = values)) +
          geom_bar(stat = "identity", fill = col1[9]) +
          #ylim(0,55) +
          theme_bw() +
          geom_text(aes(label = values), vjust = -0.5, color = col1[9], size = 3.5) +
          labs(x = " ",
               y = " ",
               title = paste0("Population: ", colony[col]))
}

p1 <- p[[1]]
p2 <- p[[2]]
p3 <- p[[3]]
p4 <- p[[4]]

png("hist1PO_all.png", width = 12, height = 4, units = "in", res = 300)

grid.arrange(p1, p2, p3, p4, ncol=4, left = textGrob("Frequency", gp=gpar(fontsize=12), vjust=1, rot=90),
             bottom = textGrob("Number of stinging bees", gp=gpar(fontsize=12), vjust=-1))
dev.off()


# create histogram like in Bees Paper
dat <- read.table("hist1_PO.txt", sep=",", fill=TRUE)
colony <- dat[1,]
colony <- colony[!is.na(colony)]
stinging <- dat[2:nrow(dat),]
dataset1 <- data.frame(dataset = 1,
                      group = c(rep("1 bee", 2), rep("2 bees", 3), 
                                rep("5 bees", 6), rep("10 bees", 11)),
                      stinging = as.factor(c(0, 1, 0:2, 0:5, 0:10)),
                      frequency = c(as.numeric(na.omit(as.vector(t(stinging))))))
dataset1$group <- factor(dataset1$group, levels=c('1 bee', '2 bees', '5 bees', 
                                                '10 bees'))
png("hist1PO_grid.png", width = 980, height = 250)
ggplot(dataset1, aes(stinging, frequency, label = stinging, width = 0.75)) + 
  geom_bar(stat="identity", fill="gray70", color="black") + 
  facet_grid(~group, scales="free", space = "free_x") +
  labs(x = " ", y = "Frequency") +
  theme_bw() +
  ggtitle("Dataset A") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))
dev.off()



### 2. experiment
dat <- read.table("hist1_IAA.txt", sep=",", fill=TRUE)
scale <- 60

# colony sizes
colony <- dat[1,]
colony <- colony[!is.na(colony)]

# number of stinging bees
stinging <- round(dat[2:nrow(dat),]*scale)

# create and save histograms for each colony size
p <- list()
for(col in 1:length(colony)){
  hist <- data.frame(bees = as.factor(c(0:colony[col])),
                     values = as.integer(stinging[col,!is.na(stinging[col,])]))
  p[[col]] <- ggplot(hist, aes(x = bees, y = values)) +
    geom_bar(stat = "identity", fill = col1[9]) +
    #ylim(0,55) +
    theme_bw() +
    geom_text(aes(label = values), vjust = -0.5, color = col1[9], size = 3.5) +
    labs(x = " ",
         y = " ",
         title = paste0("Population: ", colony[col]))
}

p1 <- p[[1]]
p2 <- p[[2]]
p3 <- p[[3]]
p4 <- p[[4]]

png("hist1IAA_all.png", width = 12, height = 4, units = "in", res = 300)

grid.arrange(p1, p2, p3, p4, ncol=4, left = textGrob("Frequency", gp=gpar(fontsize=12), vjust=1, rot=90),
             bottom = textGrob("Number of stinging bees", gp=gpar(fontsize=12), vjust=-1))
dev.off()


# create histogram like in Bees Paper
dat <- read.table("hist1_IAA.txt", sep=",", fill=TRUE)
colony <- dat[1,]
colony <- colony[!is.na(colony)]
stinging <- dat[2:nrow(dat),]
dataset2 <- data.frame(dataset = 2,
                       group = c(rep("1 bee", 2), rep("2 bees", 3), 
                                rep("5 bees", 6), rep("10 bees", 11)),
                      stinging = as.factor(c(0, 1, 0:2, 0:5, 0:10)),
                      frequency = c(as.numeric(na.omit(as.vector(t(stinging))))))
dataset2$group <- factor(dataset2$group, levels=c('1 bee', '2 bees', '5 bees', 
                                                '10 bees'))

png("hist1IAA_grid.png", width = 980, height = 250)
ggplot(dataset2, aes(stinging, frequency, label = stinging, width = 0.75)) + 
  geom_bar(stat="identity", fill="gray70", color="black") + 
  facet_grid(~group, scales="free", space = "free_x") +
  labs(x = " ", y = "Frequency") +
  theme_bw() +
  ggtitle("Dataset B") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))
dev.off()




### 3. experiment
dat <- read.table("hist2.txt", sep=",", fill=TRUE, col.names = as.character(c(0:15)))
scale <- c(68,68,60,56,52,48)

# colony sizes
colony <- dat[1,]
colony <- colony[!is.na(colony)]

# number of stinging bees
stinging <- data.frame()
for(i in 2:nrow(dat)){
  s <- round(dat[i,] * scale[(i-1)])
  stinging <- rbind(stinging, s)
}


# create and save histograms for each colony size
p <- list()
for(col in 1:length(colony)){
  hist <- data.frame(bees = as.factor(c(0:colony[col])),
                     values = as.integer(stinging[col,!is.na(stinging[col,])]))
  p[[col]] <- ggplot(hist, aes(x = bees, y = values)) +
    geom_bar(stat = "identity", fill = col1[9]) +
    #ylim(0,50) +
    theme_bw() +
    geom_text(aes(label = values), vjust = -0.5, color = col1[9], size = 3.5) +
    labs(x = " ",
         y = " ",
         title = paste0("Population: ", colony[col]))
}

p1 <- p[[1]]
p2 <- p[[2]]
p3 <- p[[3]]
p4 <- p[[4]]
p5 <- p[[5]]
p6 <- p[[6]]

png("hist2_all.png", width = 12, height = 4, units = "in", res = 300)

grid.arrange(p1, p2, p3, p4, p5, p6, ncol=6, left = textGrob("Frequency", gp=gpar(fontsize=12), vjust=1, rot=90),
             bottom = textGrob("Number of stinging bees", gp=gpar(fontsize=12), vjust=-1))
dev.off()

# create histogram like in Bees Paper
dat <- read.table("hist2.txt", sep=",", fill=TRUE)
colony <- dat[1,]
colony <- colony[!is.na(colony)]
stinging <- dat[2:nrow(dat),]
dataset3 <- data.frame(dataset = 3,
                       group = c(rep("1 bee", 2), rep("2 bees", 3), 
                                rep("5 bees", 6), rep("7 bees", 8), 
                                rep("10 bees", 11), rep("15 bees", 16)),
                      stinging = as.factor(c(0, 1, 0:2, 0:5, 0:7, 0:10, 0:15)),
                      frequency = c(as.numeric(na.omit(as.vector(t(stinging))))))

dataset3$group <- factor(dataset3$group, levels=c('1 bee', '2 bees', '5 bees', 
                                                '7 bees', '10 bees', '15 bees'))
png("hist2_grid.png", width = 980, height = 250)
ggplot(dataset3, aes(stinging, frequency, label = stinging, width = 0.75)) + 
  geom_bar(stat="identity", fill="gray70", color="black") + 
  facet_grid(~group, scales="free", space = "free_x") +
  labs(x = "Number of bees stinging", y = "Frequency") +
  theme_bw() +
  ggtitle("Dataset C") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))
dev.off()

dataset <- rbind(dataset1, dataset2, dataset3)

#png("hist1IAA_grid.png", width = 980, height = 250)
ggplot(dataset, aes(stinging, frequency, label = stinging, width = 0.75)) + 
  geom_bar(stat="identity", fill = "gray70", color = "black") + 
  #facet_grid(dataset~group, scales="free", space = "free_x") +
  facet_wrap(dataset~group, ncol = 4, dir = "h", scales = "free_x") +
  labs(x = "Number of bees stinging", y = "Frequency of observation") +
  theme_bw() +
  theme(text = element_text(size = 20))
dev.off()



# create histogram like in Bees Paper
dat <- read.table("dataset3_data.txt", sep=",", fill=TRUE)
colony <- dat[1,]
colony <- colony[!is.na(colony)]
stinging <- dat[2:nrow(dat),]
dataset1 <- data.frame(dataset = 1,
                       group = c(rep("1 bee", 2), rep("2 bees", 3), 
                                 rep("5 bees", 6), rep("10 bees", 11)),
                       stinging = as.factor(c(0, 1, 0:2, 0:5, 0:10)),
                       frequency = c(as.numeric(na.omit(as.vector(t(stinging))))))
dataset1$group <- factor(dataset1$group, levels=c('1 bee', '2 bees', '5 bees', 
                                                  '10 bees'))
png("histDS3_grid.png", width = 980, height = 250)
ggplot(dataset1, aes(stinging, frequency, label = stinging, width = 0.75)) + 
  geom_bar(stat="identity", fill="gray70", color="black") + 
  facet_grid(~group, scales="free", space = "free_x") +
  labs(x = " ", y = "Frequency") +
  theme_bw() +
  ggtitle("Dataset B") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))
dev.off()