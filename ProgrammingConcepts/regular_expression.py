def get_html_tags(html):
    """(str) --> list of str
    
    Returns list of strings containing only the html tags in the string 
    "html". Strips the   text between the tags and the tag symbols (<, >)
    
    >>> get_html_tags('<foo>asd<bar>alksjd</bar><p>asldkj</p></foo>')
    ['foo','bar', '/bar', 'p', '/p', '/foo']
    >>> get_html_tags('<a><b><c></c></b></a>')
    ['a', 'b', 'c', '/c', '/b', '/a']
    """
    import re
    
    # get the text between "<" and ">" from the string and return it as a 
    # list of strings
    html_tag = re.compile(r'<(/?\w+)>')
    return html_tag.findall(html)
    
def valid_html(test_strings):
    """ list of str --> list of tuple
    
    Returns a list of tuple. One tuple for every string in test_string 
    with first element as teh string and second element as True if 
    string has valid html tags and False if invalid html tags.
    
    >>> valid_html(['''<a><b><c></c></b></a>'''])
    [('<a><b><c></c></b></a>', True)]
    """
    # initialize list result
    result = []
    
    for item in test_strings:  
        # get html tags
        tags = get_html_tags(item) 
        
        # if the number of html tags is less than 2, it means that the html
        # string is invalid. In that case append the result False for item and
        # move to next item
        if len(tags) == 0 or len(tags) == 1:
            result.append((item, False))
            break
        
        # initialize flag and set to True
        flag = True     
        while (len(tags) > 0 and flag == True):
            flag = False    # set flag to False
            
            # if the first tag has "/" go out of the while loop without 
            # updating the flag
            if tags[0][0] == '/':
                break
            
            # compare the html tag with "/" to the previous one in the list
            # if they are same set flag to True and remove the two tags
            for i in range(len(tags)):
                if tags[i][0] == '/':
                    if tags[i-1] == tags[i][1:]:
                        flag = True
                        tags.pop(i)
                        tags.pop(i-1)
                        break
        result.append((item, flag))
    return result

# Test data
#test_strings = ['''<a><b><c></c></b></a>''',
# '''<foo>asd<bar>alksjd</bar><p>asldkj</p></foo>''',
# '''<foo><bar></bop></bar></foo>''',
# '''<foo><bar></bar></foo></foo>''',
# '''<foo><bar></foo></bar>''']

#print(valid_html(test_strings))

'''
Result should look like:
[('<a><b><c></c></b></a>', True),
 ('<foo>asd<bar>alksjd</bar><p>asldkj</p></foo>', True),
 ('<foo><bop><bar></bop></bar></foo>', False),
 ('<foo><bar></bar></foo></foo>', False),
 ('<foo><bar></foo></bar>', False)
]
'''
