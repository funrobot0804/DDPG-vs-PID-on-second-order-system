"# DDPG-vs-PID-on-second-order-system"


為了增強系統強健性系統的 
K, tau_m, zeta  
以及輸入訊號源
omega_freq amplitude offset 

都會加入一定範圍 noise 進行訓練，一次加入過多變量容易造成訓練後的網路不容易收斂。
為了收斂因此分段加入變因有利於訓練後的網路收斂
