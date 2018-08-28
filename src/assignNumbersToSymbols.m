function [ temp ] = assignNumbersToSymbols( stockData )
%ASSIGNNUMBERSTOSYMBOLS Summary of this function goes here
%   Detailed explanation goes here
    for i = 1:length(stockData)
        if isequal( stockData(i,2),{'AA'}); stockData{i,2} = '1';
        elseif isequal (stockData(i,2),{'AXP'}); stockData{i,2} = '2';
        elseif isequal (stockData(i,2),{'BA'}); stockData{i,2} = '3';
        elseif isequal (stockData(i,2),{'BAC'}); stockData{i,2} = '4';
        elseif isequal (stockData(i,2),{'CAT'}); stockData{i,2} = '5';
        elseif isequal (stockData(i,2),{'CSCO'}); stockData{i,2} = '6';
        elseif isequal (stockData(i,2),{'CVX'}); stockData{i,2} = '7';
        elseif isequal (stockData(i,2),{'DD'}); stockData{i,2} = '8';
        elseif isequal (stockData(i,2),{'DIS'}); stockData{i,2} = '9';
        elseif isequal (stockData(i,2),{'GE'}); stockData{i,2} = '10';
        elseif isequal (stockData(i,2),{'HD'}); stockData{i,2} = '11';
        elseif isequal (stockData(i,2),{'HPQ'}); stockData{i,2} = '12';
        elseif isequal (stockData(i,2),{'IBM'}); stockData{i,2} = '13';
        elseif isequal (stockData(i,2),{'INTC'}); stockData{i,2} = '14';
        elseif isequal (stockData(i,2),{'JNJ'}); stockData{i,2} = '15';
        elseif isequal (stockData(i,2),{'JPM'}); stockData{i,2} = '16';
        elseif isequal (stockData(i,2),{'KRFT'}); stockData{i,2} = '17';
        elseif isequal (stockData(i,2),{'KO'}); stockData{i,2} = '18';
        elseif isequal (stockData(i,2),{'MCD'}); stockData{i,2} = '19';
        elseif isequal (stockData(i,2),{'MMM'}); stockData{i,2} = '20';
        elseif isequal (stockData(i,2),{'MRK'}); stockData{i,2} = '21';
        elseif isequal (stockData(i,2),{'MSFT'}); stockData{i,2} = '22';
        elseif isequal (stockData(i,2),{'PFE'}); stockData{i,2} = '23';
        elseif isequal (stockData(i,2),{'PG'}); stockData{i,2} = '24';
        elseif isequal (stockData(i,2),{'T'}); stockData{i,2} = '25';
        elseif isequal (stockData(i,2),{'TRV'}); stockData{i,2} = '26';
        elseif isequal (stockData(i,2),{'UTX'}); stockData{i,2} = '27';
        elseif isequal (stockData(i,2),{'VZ'}); stockData{i,2} = '28';
        elseif isequal (stockData(i,2),{'WMT'}); stockData{i,2} = '29';
        elseif isequal (stockData(i,2),{'XOM'}); stockData{i,2} = '30';
        end;
    end;
    temp = stockData;

end