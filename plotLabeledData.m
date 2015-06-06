function plotLabeledData (XtestGrid, ypredGrid)

    gscatter (XtestGrid(:,1),XtestGrid(:,2),ypredGrid(:,1),'rbg','+*x');
    
end