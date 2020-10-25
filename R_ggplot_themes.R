# #######################
# Author: Kyle Gao 
# Sept 2020
# @Shanghai

# #######################


————# group 
x = rep(x,4)
mi= c(rep(1,10),rep(0.5,10),rep(1.5,10),rep(2,10))
y= x^mi
dt <- data.frame(x=x, y=y, mi=mi)
dt1 <- dt[c('x','y')]
ggplot(dt, aes(x=x, y=y#,
                # group= factor(mi),
                # #size= (mi),
                # shape= factor(mi) 
               ),
       alpha= 1)+
  geom_point(data = dt1, colour = "grey70")+  # add a grey background . 
  geom_line(aes(color =factor(mi))) +
  geom_point(aes(color =factor(mi))) +
  facet_grid(~mi) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.title = element_text(size = rel(0.85)), # axis labels
        plot.title = element_text(lineheight=1, hjust=0.5,  size = rel(1))
  )+
  labs(x = "rk", y = 'exponential', title ='title',
       fill=NULL ,color=NULL #legend.title
  )


———— # to make a plot just like original plot()

p <- ggplot(ds0, aes(x= x_ ))+ 
  geom_point(aes(y= y1, color='y1'))+
  geom_point(aes(y= y2, color='y2'))
lgds= c( # name = LegendsLabels 
  'y1'= 'y11', 
  'y2'= 'y222'
)
#names(lgds) <- lgds
clrs= gg_color_hue(names(lgds))
x_lbs <- seq(0, length(y),by = 50)
lucency=1
title='y1 and y2'
p + theme(text= element_text(family ="STXihei" ),
          panel.background = element_blank(), plot.background = element_blank(),
          panel.border = element_rect(fill = NA  #colour = 'black'#,linetype = "dashed"
          ),
          # panel.grid.minor = element_blank(),
          panel.grid.minor = element_line(colour = 'grey90'), 
          panel.grid.major = element_line(colour = 'grey90'),
          panel.grid.major.y = element_blank(),
          
          legend.box.background = element_rect(), # decide the border
          legend.box.margin = margin(0, 3, 0, 0), # t- r- b- l
          legend.title = element_text(size = rel(1)), # element_blank()  # then don’t display the legend.title
          legend.position = 'right', legend.justification = c(0 ,1), # inside # default "right"
          legend.background = element_rect(fill = "transparent", colour = "NA"),
          legend.key = element_rect(fill = "transparent", colour = "NA"),
          # axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          axis.title = element_text(size = rel(0.85)), # axis labels
          plot.title = element_text(lineheight=1, hjust=0.5,  size = rel(1))
)+ #end of theme 
  # scale_x_date(labels = date_format("%b-%d"),date_minor_breaks = "1 day")+ #%Y-%m-%d
  scale_x_continuous(breaks = x_lbs #,
                     # limits = c(0,200)
  ) +
  #scale_y_continuous(limits = c(0,1))+
  # scale_x_discrete(breaks = x_lbs, limits =c(0,200))+
  scale_color_manual( breaks=  names(lgds), values= clrs , labels= lgds)+
  # scale_fill_manual(breaks=brks, values=colr, labels= lbls)+
  labs(x = "rk", y = 'label no name', title =title,
       fill=NULL ,color=NULL #legend.title
  )+
  #geom_vline(xintercept = as.numeric(vline), linetype="dotted", 
  #            color = "tomato")+
  guides(fill = guide_legend(override.aes = list(alpha = 1))) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))
ggsave(paste0(format(Sys.Date(),'%y%m%d_'),title,'.png'),  path='~/wd/Rproj/R_Graphics_Output/', bg = "transparent")

191106

#   geom_point(aes( size=avgord1d), alpha=1/7) +
  # +   scale_size('日均单量',breaks=seq(0,700,by=100), range = c(0,6))+ # 气泡图

# To change plot order of facet wrap, change the order of varible levels with factor()
ds$flag <- factor(ds$flag, levels = c("S", "A", "B", "C", "SIG", "GKA", "FML”))

# facet_labels 
facet_grid(
    yfacet~xfacet,
    labeller = labeller(
        yfacet = c(`0` = "an y label", `1` = "another y label"),
        xfacet = c(`10` = "an x label", `20` = "another x label)
    )
)


GEOM_BAR    😀
★

geom_histogram(binwidth = 1, 
           position = position_dodge(width=NULL),
           stat="identity", alpha=1) 

geom_bar(width = 1, 
           position = position_dodge(width=NULL),
           stat="identity", alpha=1) 

geom_histogram 和 geom_bar 画出来的图是一样的




ds$type <- factor(ds$type, levels = c("1normal","dodge_peak","open_late","quit_early","no"))
ds$tag <- factor(ds$tag, levels = c("GKA","SIG","S","A","B","C"))

type_label <- c("1normal"='高峰期完全正常营业',
                "dodge_peak"='高峰期先关店后开店',
                "open_late"='高峰期延迟开始营业',
                "quit_early"='高峰期提前结束营业',
                "no"='高峰期"完全不营业"'
                     )
lgds= c( # name = LegendsLabels 
  'lun'='午高峰',
  'sup'='晚高峰'
)
# names(lgds) <- lgds
ds_normal <- ds[which(ds$type=="1normal"),]
ds_abn <- ds[which(ds$type!="1normal"),]
nrow(ds_abn)

ds <- ds_normal
p <- ggplot(ds,aes(x=days , y=ratio, group=peak,fill=peak 
))+
  geom_bar(width=1, position = position_dodge(width=0.8),stat="identity", alpha=1)+ # dodge 
  scale_fill_manual(breaks=  names(lgds), values= gg_color_hue(names(lgds)) , labels= lgds)+
  theme(text= element_text(family ="STXihei" ))+
  #  scale_y_continuous(   limits = c(0,0.6)    ) +
  labs(y = "商户数量占比",  x = 'occured days in 7.1-7.30', #title =paste0('days distribution',''),
       fill='' ,color='' #legend.title
  )+
  facet_wrap(~ type, nrow =length(unique(ds$type))/2
            , labeller = labeller(type= type_label)
)#+guides(fill = FALSE)



ggplot(ds,aes(x=invl_m , y=rate, group=type, width=.75,#colour=type ,
                   fill=type
                   ))+
  geom_bar(position =position_dodge(width = NULL),stat="identity", alpha=1)+
geom_bar(width=0.4, position = position_dodge(width=0.5),stat="identity", alpha=1) # dodge 
   facet_wrap(~ grp, nrow =4
             # , labeller = type+as_labeller(lgds)
              )

p <-   ggplot(ds,aes(x= timesno,y=ratio, fill=tag))+
    geom_bar(position = 'dodge',stat="identity", )+
# bar 或者 histogram 需要把 数据 rbind 后才能画出 dodge 效果。否则就是叠在一起的（并非 stack）。想想为什么。 
  # geom_bar(aes(y=ratio, fill= 'ratio'), position = 'dodge',group=1, stat="identity" )+ 
  # geom_bar(aes(y=shopratio, fill='shop',position = 'dodge'),group=2, stat="identity" )+
  facet_wrap(~restaurant_flag+respb_scene)



#EG6 :

p <- ggplot(ds, aes(x=avgcnt))+
  geom_line(aes(y=cml_ratiocnt, colour='cml_cnt'), alpha= lucency)+
  geom_line(aes(y=cml_ratio311, colour='cml_311'), alpha= lucency)+
  geom_line(aes(y=cml_ratio312, colour='cml_312'), alpha= lucency)+
  facet_wrap(~flag)

lgds= c( # name = LegendsLabels 
        'cml_311'= 'cml_3-1-1', 
        'cml_312'= 'cml_3-1-2',
        'cml_cnt'= 'cml_cnt'
        )
#names(lgds) <- lgds
clrs= gg_color_hue(names(lgds))

lucency=1
title='distributions of 311/312'

p + theme(text= element_text(family ="STXihei" ),
          # panel.background = element_blank(),plot.background = element_blank(),
          # # panel.grid.minor = element_blank(),
          # panel.grid.minor = element_line( colour = 'grey90' ), panel.grid.major = element_line( colour = 'grey90' ),
          # panel.grid.major.y = element_blank(),
          legend.title = element_text(size = rel(1)), # element_blank()  # then don’t display the legend.title
          # legend.position = c(0.8,0.5), legend.justification = c(0 ,0.5), # inside # default "right"
          # legend.background = element_rect(fill = "transparent", colour = "NA"),
          # legend.key = element_rect(fill = "transparent", colour = "NA"),
          # axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          axis.title = element_text(size = rel(0.85)), # axis labels
          plot.title = element_text(lineheight=1, hjust=0.5,  size = rel(1))
)+ #end of theme 
  # scale_x_date(labels = date_format("%b-%d"),date_minor_breaks = "1 day")+ #%Y-%m-%d
  scale_x_continuous(#breaks = x_lbs ,
                     limits = c(0,200)
                     ) +
  #scale_y_continuous(limits = c(0,1))+
  # scale_x_discrete(breaks = x_lbs, limits =c(0,200))+
  scale_color_manual( breaks=  names(lgds), values= clrs , labels= lgds )+
  # scale_fill_manual(breaks=brks, values=colr, labels= lbls)+
  labs(x = "日均订单量", y = '累计占比', title =title,
       fill=NULL ,color=NULL #legend.title
  )+
  geom_vline(xintercept = as.numeric(vline), linetype="dotted", 
             color = "tomato")+
  guides(fill = guide_legend(override.aes = list(alpha = 1))) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))


ggsave(paste0(format(Sys.Date(),'%y%m%d_'),title,'.png'),  path='~/desktop/R Graphics/', bg = "transparent")





# EG5：

p+ theme(text= element_text(family ="STXihei" ),
         panel.background = element_blank(),
         legend.title = element_text(size = rel(0.9)),
         axis.text.y=element_blank(), axis.ticks.y=element_blank(),
         axis.text.x = element_text(angle = 0, hjust = 0.5),
         plot.title = element_text(lineheight=1, hjust=0.5,  size = rel(1)))+
  #scale_x_date(labels = date_format("%Y-%m"),date_minor_breaks = "2 week")+
  scale_x_continuous(breaks = x_lbs ,limits = c(0,5400) ) + 
  # scale_x_discrete(breaks = x_lbs)+
  # scale_color_manual( breaks=brks, values=colrs)+ 
  scale_fill_manual( values='turquoise4')+
  labs(x = "mins", y = NULL, title = NULL
       ,fill=NULL ,color=NULL #legend.title
  ) +
  guides(fill = guide_legend(override.aes = list(alpha = 1))) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))





#eg4 

p <- ggplot(ds,aes(x=diff10s))+
  geom_area(aes(y=(cnt/total) ,fill='ratio'), alpha= 1) #  +
  #geom_line(aes(y=cmlcnt/total, color='cml_ratio'), alpha= lucency)

x=unique(ds$diff10s)
x_lbs <- x[seq(1, length(x),by=60*6)]
names(x_lbs) <- paste0(x_lbs/360,'')
# scale_color_manual( breaks=brks, values=colrs)+ 
# scale_fill_manual( values=colrs)

title="重复下单时间间隔分布"# "重复下单时间间隔累计分布"

p+ theme(text= element_text(family ="STXihei" ),
         legend.title = element_text(size = rel(0.9)),
         axis.text.x = element_text(angle = 0, hjust = 0.5),
         plot.title = element_text(lineheight=1, hjust=0.5,  size = rel(1)))+
  #scale_x_date(labels = date_format("%Y-%m"),date_minor_breaks = "2 week")+
  scale_x_continuous(breaks = x_lbs  ) +  
  labs(x = "Hour", y = "rate", title = title
       ,fill=NULL 
       ,color=NULL #legend.title
  ) +
  guides(fill = guide_legend(override.aes = list(alpha = 1))) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))


ggsave(paste0(title,'.png'),  path='~/desktop/R Graphics/')




#EG 3

brks = unique(ds$abn)
  colrs=rainbow(6) #brewer.pal(8,'Set1')
  names(colrs)=brks
  ggplot(ds,aes(x=order_date, y=ratio,group= abn , color=abn  ))+
    scale_x_date(labels = date_format("%Y-%m"),date_minor_breaks = "1 week")+
    theme(text= element_text(family ="STXihei" ),
          legend.title = element_text(size = rel(0.9)),
          axis.text.x = element_text(angle = 0, hjust = 1),
          plot.title = element_text(lineheight=1, hjust=0.5,  size = rel(1)))+
    scale_color_manual("异常类型",breaks=brks, values= colrs)+
    geom_line()+
    #facet_wrap(~respb_scene)+
    labs(x = "", y = "", title = '历史异常趋势')
# 加对比  
  p +geom_line( data=dcs, aes(x= order_date, y= tsp*30/max(tsp) ,color="tsp"),
                alpha=1/2,linetype='dotdash'

                )
—eg:
 colr=brewer.pal(5,'Set1')
 ggplot(x,aes(x=order_date,y=cnt,group=ottimes, color=ottimes))+
  theme(axis.text.x = element_text(angle = 60, hjust = 1))+
  scale_x_date(labels = date_format("%Y-%m"),date_minor_breaks = "1 week")+
  scale_color_gradientn(colours = cold)+
  geom_line()+
  facet_wrap(~restaurant_flag)+
  labs(x = "", y = "cnt",   title = unique(x$respb_scene))

