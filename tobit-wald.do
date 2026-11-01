regress pd y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11
estat vif
regress nd y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11
estat vif

tobit pd y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11,ll(0) ul(4) 
est store m1   
tobit nd y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11,ll(0) ul(4)
est store m2 
suest m1 m2   

testnl abs([m1_pd]y1)=abs([m2_nd]y1)   
testnl abs([m1_pd]y2)=abs([m2_nd]y2)
testnl abs([m1_pd]y3)=abs([m2_nd]y3)
testnl abs([m1_pd]y4)=abs([m2_nd]y4)
testnl abs([m1_pd]y5)=abs([m2_nd]y5)
*testnl abs([m1_pd]y6)=abs([m2_nd]y6)
testnl abs([m1_pd]y7)=abs([m2_nd]y7)
testnl abs([m1_pd]y8)=abs([m2_nd]y8)
testnl abs([m1_pd]y9)=abs([m2_nd]y9)
testnl abs([m1_pd]y10)=abs([m2_nd]y10)
testnl abs([m1_pd]y11)=abs([m2_nd]y11)
