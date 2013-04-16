source('questions.m');
source('impurity.m');
source('tree.m');

function[data] = preprocess_train(d)
    data = struct("x", d.Xtrain(1:5:3000, 1), "y", d.ytrain(1:5:3000, :));
endfunction

function[tree] = train(d)
    data = preprocess_train(d);
    tree = decisiontree(data, 0);
endfunction

function[tree] = decisiontree(node, depth)
    numpoints = size(node.y)(1);
    choices = sum(node.y);
    printf('Depth: %d\n', depth);
    if numpoints > 5 && choices != 0 && choices != numpoints && depth < 5
        [question, s] = decide(node);
        left = decisiontree(s.left, depth + 1);
        right = decisiontree(s.right, depth + 1);
        tree = make_node(@(c) split(c, question), left, right);
    else
        leaf = make_leaf(node);
        tree = leaf;
    endif
endfunction

function[question, split] = decide(node)
    imp = impurity(node);
    num = size(node.x)(1);
    features = size(node.x)(2);
    best = -1;
    for i = 1:features
        printf('Sorting feature: %d\n', i);
        [d, index] = sort(node.x(:, i));
        data = node.x(index,:);
        for j = 1:num
            split_value = data(j, i);
            q = strcat(['c.x(:, ', num2str(i),') >= ',num2str(split_value)]);
            [s, gain] = choose(node, q);
            s
            if gain > best
                best = gain;
                split = s;
                question = q;
            endif
        endfor
    endfor
    best = -1;
endfunction

function[split, gain] = choose(node, question)
    num = size(node.x)(1);
    q = @(c) split(c, question);
    split = q(node);
    inf = impurity(node);
    s = size(split.left.x);
    p = (s(1))/num;
    gain = inf - p*impurity(split.left)-(1-p)*impurity(split.right);
endfunction

function[questions] = copyquestions(qs)
    s = size(qs)(1);
    questions = cell(s,1);
    for i = 1:s
        questions{i} = qs{i};
    endfor
endfunction
