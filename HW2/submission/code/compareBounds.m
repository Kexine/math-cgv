function result = compareBounds(lower1, upper1, lower2, upper2)

if upper1 > upper2
    result = true;
elseif upper1 == upper2
    if lower1 > lower2
        result = true;
    else
        result = false;
    end
else
    result = false;
end

end