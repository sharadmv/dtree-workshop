function[data] = preprocess_test(d)
    data = struct("x", d.Xtest, "y", d.ytest);
endfunction

function[err] = test(d, tree, questions)
    data = preprocess_test(d);
    num_test = size(data.x)(1);
    num_features = size(data.x)(2);
    for i = 1:num_test
        val = classify(data.x(i,:), tree)
    endfor
endfunction

function[val] = ask(vector, question)
    val = size(question(vector).left.x)(1);
endfunction

function[val, posterior] = classify(vector, tree)
    if is_node(tree)
        a = ask(vector, get_question(tree))
        if a
            [val, posterior] = classify(vector, get_left(tree));
        else
            [val, posterior] = classify(vector, get_right(tree));
        endif
    else
        vals
        vals = get_values(tree);
        num = size(vals.y)(1);
        prob = sum(vals.y)/num;
        val = prob > 0.5;
        if val
            posterior = prob;
        else
            posterior = 1 - prob;
        endif
    endif
endfunction
