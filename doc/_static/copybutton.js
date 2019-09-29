// Copyright 2014 PSF. Licensed under the PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
// File originates from the cpython source found in Doc/tools/sphinxext/static/copybutton.js

// http://forum.jquery.com/topic/beginner-function-toggle-deprecated-what-to-use-instead#14737000003772143
$.fn.clicktoggle = function(a, b) {
    return this.each(function() {
        var clicked = false;
        $(this).click(function() {
            if (clicked) {
                clicked = false;
                return b.apply(this, arguments);
            }
            clicked = true;
            return a.apply(this, arguments);
        });
    });
};

$(document).ready(function() {
    /* Add a [>>>] button on the top-right corner of code samples to hide
     * the >>> and ... prompts and the output and thus make the code
     * copyable. */
    var div = $('.highlight-default .highlight,' +
                '.highlight-python .highlight' +
                '.highlight-python3 .highlight')
    var pre = div.find('pre');

    // get the styles from the current theme
    pre.parent().parent().css('position', 'relative');
    var hide_text = 'Hide the prompts and output';
    var show_text = 'Show the prompts and output';
    var border_width = pre.css('border-top-width');
    var border_style = pre.css('border-top-style');
    var border_color = pre.css('border-top-color');
    var button_styles = {
        'cursor':'pointer', 'position': 'absolute', 'top': '0', 'right': '0',
        'border-color': border_color, 'border-style': border_style,
        'border-width': border_width, 'color': border_color, 'text-size': '75%',
        'font-family': 'monospace', 'padding-left': '0.2em', 'padding-right': '0.2em',
        'border-radius': '0 3px 0 0'
    }

    // create and add the button to all the code blocks that contain >>>
    div.each(function(index) {
        var jthis = $(this);
        if (jthis.find('.gp').length > 0) {
            var button = $('<span class="copybutton">&gt;&gt;&gt;</span>');
            button.css(button_styles);
            button.attr('title', hide_text);
            jthis.prepend(button);
        }
        // tracebacks (.gt) contain bare text elements that need to be
        // wrapped in a span to work with .nextUntil() (see later)
        jthis.find('pre:has(.gt)').contents().filter(function() {
            return ((this.nodeType == 3) && (this.data.trim().length > 0));
        }).wrap('<span>');
    });

    // define the behavior of the button when it's clicked
    $('.copybutton').clicktoggle(
        function() {
            var button = $(this);
            button.parent().find('.go, .gp, .gt').hide();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'hidden');
            button.css('text-decoration', 'line-through');
            button.attr('title', show_text);
        },
        function() {
            var button = $(this);
            button.parent().find('.go, .gp, .gt').show();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'visible');
            button.css('text-decoration', 'none');
            button.attr('title', hide_text);
        });
});


// "Responsive side bar
$(document).ready(function() {

   (function($) {
     $(function() {
       $('body').scrollspy({
         target: '#sidebar',
         offset: 60
       });

       var cur_path = paths(location.pathname);
       var links = $("#navbar ul li a");
       var max_length = -1;
       var pos = -1;
       for (var i = 0; i < links.length; i++) {
         if (links[i].getAttribute("href") === "#")
           continue;
         // Ignore external links
         if (links[i].host !== location.host)
           continue;

         var nav_path = paths(links[i].pathname);

         var length = prefix_length(nav_path, cur_path);
         if (length > max_length) {
           max_length = length;
           pos = i;
         }
       }

       // Add class to parent <li>, and enclosing <li> if in dropdown
       if (pos >= 0) {
         var menu_anchor = $(links[pos]);
         menu_anchor.parent().addClass("active");
         menu_anchor.closest("li.dropdown").addClass("active");
       }
     });

     function paths(pathname) {
       var pieces = pathname.split("/");
       pieces.shift(); // always starts with /

       var end = pieces[pieces.length - 1];
       if (end === "index.html" || end === "")
         pieces.pop();
       return(pieces);
     }

     // Returns -1 if not found
     function prefix_length(needle, haystack) {
       if (needle.length > haystack.length)
         return(-1);

       // Special case for length-0 haystack, since for loop won't run
       if (haystack.length === 0) {
         return(needle.length === 0 ? 0 : -1);
       }

       for (var i = 0; i < haystack.length; i++) {
         if (needle[i] != haystack[i])
           return(i);
       }

       return(haystack.length);
     }
   })(window.$jqTheme || window.jQuery)
})
