function [hhhh] = generateHist0(a,b)

  
    max_a = max(a);
    hhhh=max(a);
    max_b = max(b);
    max_together = max(max_a,max_b);
    stepLength=60;
    rightEdge = ceil(max_together/stepLength)*stepLength;
    edges_together = 0:stepLength:rightEdge;
    h_figure = figure;
    set(h_figure,'units','normalized','position',[0.1 0.1 0.8 0.7]);
    
    ax1 = subplot(2,2,1);
    h = histogram(a,edges_together,'FaceAlpha',0.5);
    hold on;
    values1 = h.Values;
    edges1 = h.BinEdges;
    binwidth1 = h.BinWidth;
    centerpoints1(1,:) = edges1(1:end-1)+0.5*binwidth1;
    centerpoints1(2,:) = values1;
    plot(centerpoints1(1,:),centerpoints1(2,:),'--r*','MarkerSize',8,'MarkerEdgeColor','m','MarkerFaceColor','m');

    lgd=legend('理论时间','理论时间分布曲线');
    hold off;
    
    ax2 = subplot(2,2,2);
    h = histogram(b,edges_together,'FaceAlpha',0.5,'FaceColor',[1.0 0.5 0.4]);
    hold on;
    values2 = h.Values;
    edges2 = h.BinEdges;
    binwidth2 = h.BinWidth;
    centerpoints2(1,:) = edges2(1:end-1)+0.5*binwidth2;
    centerpoints2(2,:) = values2;
    plot(centerpoints2(1,:),centerpoints2(2,:),'--b*','MarkerSize',8,'MarkerEdgeColor','m','MarkerFaceColor','m');
    lgd2=legend('实际时间','实际时间分布曲线');
    hold off;
    
    max_value = max(max(values1),max(values2));
    ystep = floor((max_value+2)/10);
    ylim(ax1,[0 max_value+2])
    ytick = (max_value+2);
    debug = [0:ystep:max_value,max_value:ystep:max_value+2];
    yticks(ax1,[0:ystep:max_value+2]);
    ylim(ax2,[0 max_value+2])
    ytick = (max_value+2);
    yticks(ax2,[0:ystep:max_value+2]);   
    
    ax3 = subplot(2,2,3);
    h1 = histogram(a,edges_together,'FaceAlpha',0.5);
    hold on;
    h2=histogram(b,edges_together,'FaceAlpha',0.5,'FaceColor',[1.0 0.5 0.4]);
    values = h1.Values;
    edges = h1.BinEdges;
    binwidth = h2.BinWidth;
    ylim(ax3,[0 max_value+2])
    ytick = (max_value+2);
    yticks(ax3,[0:ystep:max_value+2]); 
    plot(centerpoints1(1,:),centerpoints1(2,:),'--r*','MarkerSize',8,'MarkerEdgeColor','m','MarkerFaceColor','m');
    plot(centerpoints2(1,:),centerpoints2(2,:),'--b*','MarkerSize',8,'MarkerEdgeColor','m','MarkerFaceColor','m');
    lgd=legend('理论时间','实际时间','理论时间分布曲线','实际时间分布曲线');

    minBoundary1 = min(find(values1~=0));
    minBoundary2 = min(find(values2~=0));
    minBoundary = min(minBoundary1,minBoundary2);
    
    xtick = (rightEdge+120);
    xlim(ax1,[60*(minBoundary1-1) rightEdge+120]) 
    xticks(ax1,60*(minBoundary1-1):360:xtick);
    xlim(ax2,[60*(minBoundary2-1) rightEdge+120]) 
    xticks(ax2,60*(minBoundary2-1):360:xtick);
    xlim(ax3,[60*(minBoundary-1) rightEdge+120]) 
    xticks(ax3,60*(minBoundary-1):360:xtick);
end

