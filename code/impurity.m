function[e] = impurity(node)
    if size(node.x)(1) == 0
        e = 0;
    else
        e = entropy(node);
    endif
endfunction

function[e] = entropy(node)
    num = size(node.y)(1);
    p = sum(node.y)/num;
    q = 1 - p;
    if p == 0
        p = 0.0001;
    endif
    if q == 0
        q = 0.0001;
    endif
    e = -p*log2(p)-q*log2(q);
endfunction

function[e] = gini(node)
    num = size(node.y)(1);
    p = sum(node.y)/num;
    e = 1 - p*p;
endfunction
