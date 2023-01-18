#Torch, K = 10, 10 subjects
X <- read.csv(file = '/Users/frederikkeuldahl/Desktop/Fagprojekt/GitHub/GeneralLinearModel/XTorch_B_K=10.csv')
y <- read.csv(file = '/Users/frederikkeuldahl/Desktop/Fagprojekt/GitHub/GeneralLinearModel/y_K=10.csv')

D <- data.frame(X = X,y=y$X0-1/2)

fit <- lm(y~-1+X.X0+X.X1+X.X2+X.X3+X.X4+X.X5+X.X6+X.X7+X.X8+X.X9,data = D)

summary(fit)

#HMM, K = 10, 10 subjects
X <- read.csv(file = '/Users/frederikkeuldahl/Desktop/Fagprojekt/GitHub/GeneralLinearModel/XHMM_B_K=10.csv')
y <- read.csv(file = '/Users/frederikkeuldahl/Desktop/Fagprojekt/GitHub/GeneralLinearModel/y_K=10.csv')

D <- data.frame(X = X,y=y$X0-1/2)

fit <- lm(y~-1+X.X0+X.X1+X.X2+X.X3+X.X4+X.X5+X.X6+X.X7+X.X8+X.X9,data = D)

summary(fit)



