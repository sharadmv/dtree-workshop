function[leaf] = make_leaf(node)
    leaf = cell(2,1);
    leaf{1} = 'leaf';
    leaf{2} = node;
endfunction

function[node] = get_values(leaf)
    node = leaf{2};
endfunction

function[val] = is_leaf(node)
    val = node{1} == 'leaf';
endfunction

function[node] = make_node(question, left, right)
    node = cell(2,1);
    node{1} = 'node';
    node{2} = question;
    children = struct();
    children.left = left;
    children.right = right;
    node{3} = children;
endfunction

function[val] = is_node(node)
    val = node{1} == 'node';
endfunction

function[question] = get_left(node)
    question = node{3}.left;
endfunction

function[question] = get_right(node)
    question = node{3}.right;
endfunction

function[question] = get_question(node)
    question = node{2};
endfunction
