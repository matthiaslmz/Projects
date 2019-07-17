'use strict';

var svg, tooltip, biHiSankey, path, defs, colorScale, highlightColorScale, isTransitioning;

var OPACITY = {
    NODE_DEFAULT: 0.9,
    NODE_FADED: 0.1,
    NODE_HIGHLIGHT: 0.8,
    LINK_DEFAULT: 0.6,
    LINK_FADED: 0.05,
    LINK_HIGHLIGHT: 0.9
  },
  TYPES = ["Central Elements", "Categories", "Influences"],
  TYPE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"],
  TYPE_HIGHLIGHT_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb"],
  LINK_COLOR = "#b3b3b3",
  INFLOW_COLOR = "#2E86D1",
  OUTFLOW_COLOR = "#D63028",
  NODE_WIDTH = 40,
  COLLAPSER = {
    RADIUS: NODE_WIDTH / 2,
    SPACING: 5
  },
  OUTER_MARGIN = 8,
  MARGIN = {
    TOP: 2 * (COLLAPSER.RADIUS + OUTER_MARGIN),
    RIGHT: OUTER_MARGIN,
    BOTTOM: OUTER_MARGIN,
    LEFT: OUTER_MARGIN
  },
  TRANSITION_DURATION = 400,
  HEIGHT = 650 - MARGIN.TOP - MARGIN.BOTTOM,
  WIDTH = 1200 - MARGIN.LEFT - MARGIN.RIGHT,
  LAYOUT_INTERATIONS = 32,
  REFRESH_INTERVAL = 7000;

var formatNumber = function (d) {
  var numberFormat = d3.format(",.0f"); // zero decimal places
  return numberFormat(d);
},

formatFlow = function (d) {
  var flowFormat = d3.format(",.0f"); // zero decimal places with sign
  return flowFormat(Math.abs(d));
},

// Used when temporarily disabling user interractions to allow animations to complete
disableUserInterractions = function (time) {
  isTransitioning = true;
  setTimeout(function(){
    isTransitioning = false;
  }, time);
},

hideTooltip = function () {
  return tooltip.transition()
    .duration(TRANSITION_DURATION)
    .style("opacity", 0);
},

showTooltip = function () {
  return tooltip
    .style("left", d3.event.pageX + "px")
    .style("top", d3.event.pageY + 15 + "px")
    .transition()
      .duration(TRANSITION_DURATION)
      .style("opacity", 1);
};

colorScale = d3.scale.ordinal().domain(TYPES).range(TYPE_COLORS),
highlightColorScale = d3.scale.ordinal().domain(TYPES).range(TYPE_HIGHLIGHT_COLORS),

svg = d3.select("#chart").append("svg")
        .attr("width", WIDTH + MARGIN.LEFT + MARGIN.RIGHT)
        .attr("height", HEIGHT + MARGIN.TOP + MARGIN.BOTTOM)
      .append("g")
        .attr("transform", "translate(" + MARGIN.LEFT + "," + MARGIN.TOP + ")");

svg.append("g").attr("id", "links");
svg.append("g").attr("id", "nodes");
svg.append("g").attr("id", "collapsers");

tooltip = d3.select("#chart").append("div").attr("id", "tooltip");

tooltip.style("opacity", 0)
    .append("p")
      .attr("class", "value");

biHiSankey = d3.biHiSankey();

// Set the biHiSankey diagram properties
biHiSankey
  .nodeWidth(NODE_WIDTH)
  .nodeSpacing(8)
  .linkSpacing(10 )
  .arrowheadScaleFactor(0.5) // Specifies that 0.5 of the link's stroke WIDTH should be allowed for the marker at the end of the link.
  .size([WIDTH, HEIGHT]);

path = biHiSankey.link().curvature(0.40);

defs = svg.append("defs");

defs.append("marker")
  .style("fill", LINK_COLOR)
  .attr("id", "arrowHead")
  .attr("viewBox", "0 0 6 10")
  .attr("refX", "1")
  .attr("refY", "5")
  .attr("markerUnits", "strokeWidth")
  .attr("markerWidth", "1")
  .attr("markerHeight", "1")
  .attr("orient", "auto")
  .append("path")
    .attr("d", "M 0 0 L 1 0 L 6 5 L 1 10 L 0 10 z");

defs.append("marker")
  .style("fill", OUTFLOW_COLOR)
  .attr("id", "arrowHeadInflow")
  .attr("viewBox", "0 0 6 10")
  .attr("refX", "1")
  .attr("refY", "5")
  .attr("markerUnits", "strokeWidth")
  .attr("markerWidth", "1")
  .attr("markerHeight", "1")
  .attr("orient", "auto")
  .append("path")
    .attr("d", "M 0 0 L 1 0 L 6 5 L 1 10 L 0 10 z");

defs.append("marker")
  .style("fill", INFLOW_COLOR)
  .attr("id", "arrowHeadOutlow")
  .attr("viewBox", "0 0 6 10")
  .attr("refX", "1")
  .attr("refY", "5")
  .attr("markerUnits", "strokeWidth")
  .attr("markerWidth", "1")
  .attr("markerHeight", "1")
  .attr("orient", "auto")
  .append("path")
    .attr("d", "M 0 0 L 1 0 L 6 5 L 1 10 L 0 10 z");

function update () {
  var link, linkEnter, node, nodeEnter, collapser, collapserEnter;

  function dragmove(node) {
    node.x = Math.max(0, Math.min(WIDTH - node.width, d3.event.x));
    node.y = Math.max(0, Math.min(HEIGHT - node.height, d3.event.y));
    d3.select(this).attr("transform", "translate(" + node.x + "," + node.y + ")");
    biHiSankey.relayout();
    svg.selectAll(".node").selectAll("rect").attr("height", function (d) { return d.height; });
    link.attr("d", path);
  }

  function containChildren(node) {
    node.children.forEach(function (child) {
      child.state = "contained";
      child.parent = this;
      child._parent = null;
      containChildren(child);
    }, node);
  }

  function expand(node) {
    node.state = "expanded";
    node.children.forEach(function (child) {
      child.state = "collapsed";
      child._parent = this;
      child.parent = null;
      containChildren(child);
    }, node);
  }

  function collapse(node) {
    node.state = "collapsed";
    containChildren(node);
  }

  function restoreLinksAndNodes() {
    link
      .style("stroke", LINK_COLOR)
      .style("marker-end", function () { return 'url(#arrowHead)'; })
      .transition()
        .duration(TRANSITION_DURATION)
        .style("opacity", OPACITY.LINK_DEFAULT);

    node
      .selectAll("rect")
        .style("fill", function (d) {
          d.color = colorScale(d.type.replace(/ .*/, ""));
          return d.color;
        })
        .style("stroke", function (d) {
          return d3.rgb(colorScale(d.type.replace(/ .*/, ""))).darker(0.1);
        })
        .style("fill-opacity", OPACITY.NODE_DEFAULT);

    node.filter(function (n) { return n.state === "collapsed"; })
      .transition()
        .duration(TRANSITION_DURATION)
        .style("opacity", OPACITY.NODE_DEFAULT);
  }

  function showHideChildren(node) {
    disableUserInterractions(2 * TRANSITION_DURATION);
    hideTooltip();
    if (node.state === "collapsed") { expand(node); }
    else { collapse(node); }

    biHiSankey.relayout();
    update();
    link.attr("d", path);
    restoreLinksAndNodes();
  }

  function highlightConnected(g) {
    link.filter(function (d) { return d.source === g; })
      .style("marker-end", function () { return 'url(#arrowHeadInflow)'; })
      .style("stroke", OUTFLOW_COLOR)
      .style("opacity", OPACITY.LINK_DEFAULT);

    link.filter(function (d) { return d.target === g; })
      .style("marker-end", function () { return 'url(#arrowHeadOutlow)'; })
      .style("stroke", INFLOW_COLOR)
      .style("opacity", OPACITY.LINK_DEFAULT);
  }

  function fadeUnconnected(g) {
    link.filter(function (d) { return d.source !== g && d.target !== g; })
      .style("marker-end", function () { return 'url(#arrowHead)'; })
      .transition()
        .duration(TRANSITION_DURATION)
        .style("opacity", OPACITY.LINK_FADED);

    node.filter(function (d) {
      return (d.name === g.name) ? false : !biHiSankey.connected(d, g);
    }).transition()
      .duration(TRANSITION_DURATION)
      .style("opacity", OPACITY.NODE_FADED);
  }

  link = svg.select("#links").selectAll("path.link")
    .data(biHiSankey.visibleLinks(), function (d) { return d.id; });

  link.transition()
    .duration(TRANSITION_DURATION)
    .style("stroke-WIDTH", function (d) { return Math.max(10, d.thickness); })
    .attr("d", path)
    .style("opacity", OPACITY.LINK_DEFAULT);



  link.exit().remove();


  linkEnter = link.enter().append("path")
    .attr("class", "link")
    .style("fill", "none");
  link.on("click", function(d){
  window.open(d.target_url)});
  linkEnter.on('mouseenter', function (d) {
    if (!isTransitioning) {
      showTooltip().select(".value").text(function () {
        if (d.direction > 0) {
          return d.source.name + " → " + d.target.name + "\n" + formatNumber(d.value);
        }
        return d.target.name + " ← " + d.source.name + "\n" + formatNumber(d.value);
      });

      d3.select(this)
        .style("stroke", LINK_COLOR)
        .transition()
          .duration(TRANSITION_DURATION / 2)
          .style("opacity", OPACITY.LINK_HIGHLIGHT);
    }
  });

  linkEnter.on('mouseleave', function () {
    if (!isTransitioning) {
      hideTooltip();

      d3.select(this)
        .style("stroke", LINK_COLOR)
        .transition()
          .duration(TRANSITION_DURATION / 2)
          .style("opacity", OPACITY.LINK_DEFAULT);
    }
  });

  linkEnter.sort(function (a, b) { return b.thickness - a.thickness; })
    .classed("leftToRight", function (d) {
      return d.direction > 0;
    })
    .classed("rightToLeft", function (d) {
      return d.direction < 0;
    })
    .style("marker-end", function () {
      return 'url(#arrowHead)';
    })
    .style("stroke", LINK_COLOR)
    .style("opacity", 0)
    .transition()
      .delay(TRANSITION_DURATION)
      .duration(TRANSITION_DURATION)
      .attr("d", path)
      .style("stroke-WIDTH", function (d) { return Math.max(10, d.thickness); })
      .style("opacity", OPACITY.LINK_DEFAULT);


  node = svg.select("#nodes").selectAll(".node")
      .data(biHiSankey.collapsedNodes(), function (d) { return d.id; });


  node.transition()
    .duration(TRANSITION_DURATION)
    .attr("transform", function (d) { return "translate(" + d.x + "," + d.y + ")"; })
    .style("opacity", OPACITY.NODE_DEFAULT)
    .select("rect")
      .style("fill", function (d) {
        d.color = colorScale(d.type.replace(/ .*/, ""));
        return d.color;
      })
      .style("stroke", function (d) { return d3.rgb(colorScale(d.type.replace(/ .*/, ""))).darker(0.1); })
      .style("stroke-WIDTH", "1px")
      .attr("height", function (d) { return d.height; })
      .attr("width", biHiSankey.nodeWidth());


  node.exit()
    .transition()
      .duration(TRANSITION_DURATION)
      .attr("transform", function (d) {
        var collapsedAncestor, endX, endY;
        collapsedAncestor = d.ancestors.filter(function (a) {
          return a.state === "collapsed";
        })[0];
        endX = collapsedAncestor ? collapsedAncestor.x : d.x;
        endY = collapsedAncestor ? collapsedAncestor.y : d.y;
        return "translate(" + endX + "," + endY + ")";
      })
      .remove();


  nodeEnter = node.enter().append("g").attr("class", "node");

  nodeEnter
    .attr("transform", function (d) {
      var startX = d._parent ? d._parent.x : d.x,
          startY = d._parent ? d._parent.y : d.y;
      return "translate(" + startX + "," + startY + ")";
    })
    .style("opacity", 1e-6)
    .transition()
      .duration(TRANSITION_DURATION)
      .style("opacity", OPACITY.NODE_DEFAULT)
      .attr("transform", function (d) { return "translate(" + d.x + "," + d.y + ")"; });

  nodeEnter.append("text");
  nodeEnter.append("rect")
    .style("fill", function (d) {
      d.color = colorScale(d.type.replace(/ .*/, ""));
      return d.color;
    })
    .style("stroke", function (d) {
      return d3.rgb(colorScale(d.type.replace(/ .*/, ""))).darker(0.1);
    })
    .style("stroke-WIDTH", "1px")
    .attr("height", function (d) { return d.height; })
    .attr("width", biHiSankey.nodeWidth());

  node.on("mouseenter", function (g) {
    if (!isTransitioning) {
      restoreLinksAndNodes();
      highlightConnected(g);
      fadeUnconnected(g);

      d3.select(this).select("rect")
        .style("fill", function (d) {
          d.color = d.netFlow > 0 ? INFLOW_COLOR : OUTFLOW_COLOR;
          return d.color;
        })
        .style("stroke", function (d) {
          return d3.rgb(d.color).darker(0.1);
        })
        .style("fill-opacity", OPACITY.LINK_DEFAULT);

      tooltip
        .style("left", g.x + MARGIN.LEFT + "px")
        .style("top", g.y + g.height + MARGIN.TOP + 15 + "px")
        .transition()
          .duration(TRANSITION_DURATION)
          .style("opacity", 1).select(".value")
          .text(function () {
            var additionalInstructions = g.children.length ? "\n(Double click to expand)" : "";
            return g.name + "\nNet flow: " + formatFlow(g.netFlow) + additionalInstructions;
          });
    }
  });

  node.on("mouseleave", function () {
    if (!isTransitioning) {
      hideTooltip();
      restoreLinksAndNodes();
    }
  });
  

 

  /**
  * Fix to allow for dblclick on dragging element
  * This essentially checks to see if the vectors are in the same location once the drag
  * has ended.
  */

  var lastvector = []
  function isclicked(node){
    try {
      if( lastvector[node.id].toString() !== [node.x,node.y].toString() ){
        throw 'no match';
      }
      showHideChildren(node);
    }catch(err) {
      lastvector[node.id] = [node.x,node.y]
    }
  }

  // allow nodes to be dragged to new positions
  node.call(d3.behavior.drag()
    .origin(function (d) { return d; })
    .on("dragstart", function () { node.event,this.parentNode.appendChild(this); })
    .on("dragend", isclicked)
    .on("drag", dragmove));

  // add in the text for the nodes
  node.filter(function (d) { return d.value !== 0; })
    .select("text")
      .attr("x", -6)
      .attr("y", function (d) { return d.height / 2; })
      .attr("dy", ".35em")
      .attr("text-anchor", "end")
      .attr("transform", null)
      .text(function (d) { return d.name; })
    .filter(function (d) { return d.x < WIDTH / 2; })
      .attr("x", 6 + biHiSankey.nodeWidth())
      .attr("text-anchor", "start");


  collapser = svg.select("#collapsers").selectAll(".collapser")
    .data(biHiSankey.expandedNodes(), function (d) { return d.id; });


  collapserEnter = collapser.enter().append("g").attr("class", "collapser");

  collapserEnter.append("circle")
    .attr("r", COLLAPSER.RADIUS)
    .style("fill", function (d) {
      d.color = colorScale(d.type.replace(/ .*/, ""));
      return d.color;
    });

  collapserEnter
    .style("opacity", OPACITY.NODE_DEFAULT)
    .attr("transform", function (d) {
      return "translate(" + (d.x + d.width / 2) + "," + (d.y + COLLAPSER.RADIUS) + ")";
    });

  collapserEnter.on("dblclick", showHideChildren);

  collapser.select("circle")
    .attr("r", COLLAPSER.RADIUS);

  collapser.transition()
    .delay(TRANSITION_DURATION)
    .duration(TRANSITION_DURATION)
    .attr("transform", function (d, i) {
      return "translate("
        + (COLLAPSER.RADIUS + i * 2 * (COLLAPSER.RADIUS + COLLAPSER.SPACING))
        + ","
        + (-COLLAPSER.RADIUS - OUTER_MARGIN)
        + ")";
    });

  collapser.on("mouseenter", function (g) {
    if (!isTransitioning) {
      showTooltip().select(".value")
        .text(function () {
          return g.name + "\n(Double click to collapse)";
        });

      var highlightColor = highlightColorScale(g.type.replace(/ .*/, ""));

      d3.select(this)
        .style("opacity", OPACITY.NODE_HIGHLIGHT)
        .select("circle")
          .style("fill", highlightColor);

      node.filter(function (d) {
        return d.ancestors.indexOf(g) >= 0;
      }).style("opacity", OPACITY.NODE_HIGHLIGHT)
        .select("rect")
          .style("fill", highlightColor);
    }
  });

  collapser.on("mouseleave", function (g) {
    if (!isTransitioning) {
      hideTooltip();
      d3.select(this)
        .style("opacity", OPACITY.NODE_DEFAULT)
        .select("circle")
          .style("fill", function (d) { return d.color; });

      node.filter(function (d) {
        return d.ancestors.indexOf(g) >= 0;
      }).style("opacity", OPACITY.NODE_DEFAULT)
        .select("rect")
          .style("fill", function (d) { return d.color; });
    }
  });

  collapser.exit().remove();
  
  

}


    d3.selectAll('.sankey-align').on('change', function() {
      biHiSankey.align(this.value)
            .layout(32);
      d3Digest();
    });

var exampleNodes = [{
		"type": "master",
		"id": "a",
		"parent": null,
		"name": "Central Elements",
		"url": "none"
	},
	{
		"type": "node",
		"id": 1,
		"parent": "a",
		"number": "101",
		"name": "Food production Hunting and Harvesting",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting"
	},
	{
		"type": "node",
		"id": 2,
		"parent": "a",
		"number": "120",
		"name": "Imports",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Imports"
	},
	{
		"type": "node",
		"id": 3,
		"parent": "a",
		"number": "140",
		"name": "Exports",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Exports"
	},
	{
		"type": "node",
		"id": 4,
		"parent": "a",
		"number": "150",
		"name": "Wastage, non-food use, animal food",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Wastage+non-food+use+animal+food"
	},
	{
		"type": "node",
		"id": 5,
		"parent": "a",
		"number": "160",
		"name": "Storage",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
	},
	{
		"type": "node",
		"id": 6,
		"parent": "a",
		"number": "170",
		"name": "Available food supply Traditional foods",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods"
	},
	{
		"type": "node",
		"id": 7,
		"parent": "a",
		"number": "170",
		"name": "Food  processing",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing"
	},
	{
		"type": "node",
		"id": 8,
		"parent": "a",
		"number": "180",
		"name": "Food for sale, intended to be eaten at home/eaten away from home Traditional foods",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
	},
	{
		"type": "node",
		"id": 9,
		"parent": "a",
		"number": "190",
		"name": "Food acquired",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Acquired"
	},
	{
		"type": "node",
		"id": 10,
		"parent": "a",
		"number": "110",
		"name": "Food consumed",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Consumed"
	},
	{
		"type": "node",
		"id": 11,
		"parent": "a",
		"number": "171",
		"name": "Nutrient & nonnutrient intakes",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
	},
	{
		"type": "node",
		"id": 12,
		"parent": "a",
		"number": "121",
		"name": "Nutrient & non-nutrient metabolic utilization: Nutritional, Chemical and Microbial Status",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
	},
	{
		"type": "node",
		"id": 13,
		"parent": "a",
		"number": "122",
		"name": "Supplements and natural health products",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health"
	},
	{
		"type": "node",
		"id": 14,
		"parent": "a",
		"number": "125",
		"name": "Genetic expression",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression"
	},
	{
		"type": "node",
		"id": 15,
		"parent": "a",
		"number": "128",
		"name": "Health status, outcomes morbidity, mortality",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes"
	},


	{
		"type": "slave",
		"id": "b",
		"parent": null,
		"number": "l",
		"name": "Categories",
		"url": "none"
	},
	{
		"type": "node",
		"id": 16,
		"parent": "b",
		"number": "210",
		"name": "Food Supply",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply"
	},
	{
		"type": "node",
		"id": 17,
		"parent": "b",
		"number": "215",
		"name": "Distribution",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Distribution"
	},
	{
		"type": "node",
		"id": 18,
		"parent": "b",
		"number": "220",
		"name": "Consumption",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumption"
	},
	{
		"type": "node",
		"id": 19,
		"parent": "b",
		"number": "230",
		"name": "Utilization",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Utilization"
	},
	{
		"type": "node",
		"id": 20,
		"parent": "b",
		"number": "240",
		"name": "Health Outcome",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Health+outcome"
	},


	{
		"type": "peer",
		"id": "c",
		"parent": null,
		"number": "ex",
		"name": "Influences",
		"url": "none"
	},
	{
		"type": "node",
		"id": 21,
		"parent": "c",
		"number": "500",
		"name": "Environmental and Agricultural Influences",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Environmental+and+Agricultural+Influences"

	}, {
		"type": "node",
		"id": 22,
		"parent": "c",
		"number": "510",
		"name": "Transport",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Transport"
	}, {
		"type": "node",
		"id": 23,
		"parent": "c",
		"number": "540",
		"name": "Food Composition",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Consumption"
	}, {
		"type": "node",
		"id": 24,
		"parent": "c",
		"number": "560",
		"name": "Consumer demand",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumer+demand"
	}, {
		"type": "node",
		"id": 25,
		"parent": "c",
		"number": "570",
		"name": "Access and equity issues",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Access+and+equity+issues"
	}, {
		"type": "node",
		"id": 26,
		"parent": "c",
		"number": "580",
		"name": "Education Information Advertising",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Education+Information+Advertising"
	}, {
		"type": "node",
		"id": 27,
		"parent": "c",
		"number": "590",
		"name": "Knowledge Attitudes Beliefs Behaviours",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Knowledge+Attitudes+Beliefs+Behaviours"
	}, {
		"type": "node",
		"id": 28,
		"parent": "c",
		"number": "515",
		"name": "Food Preference",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference"
	},

	{
		"type": "node",
		"id": 29,
		"parent": "c",
		"number": "535",
		"name": "Income",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=income"
	}, {
		"type": "node",
		"id": 30,
		"parent": "c",
		"number": "545",
		"name": "Price",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=price"
	}, {
		"type": "node",
		"id": 31,
		"parent": "c",
		"number": "555",
		"name": "Age, sex, physical status, activity/lifestyle, drug use, genetic characteristics",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Age+sex+physical+status"
	}, {
		"type": "node",
		"id": 32,
		"parent": "c",
		"number": "565",
		"name": "Existing health status and nutrient requirements",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Existing+health+status+and+nutrient+requirements"
	}, {
		"type": "node",
		"id": 33,
		"parent": "c",
		"number": "575",
		"name": "Housing, sanitation, occupation, services",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Housing+sanitation"
	}, {
		"type": "node",
		"id": 34,
		"parent": "c",
		"number": "585",
		"name": "Contaminants in foods",
		"url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Contaminants+in+foods"
	}

]

var exampleLinks = [
 {
   "source": 16,
   "target": 17,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Distribution"
 },
 {
   "source": 17,
   "target": 18,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Distribution",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumption"
 },
 {
   "source": 18,
   "target": 19,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumption",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Utilization"
 },
 {
   "source": 19,
   "target": 20,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Utilization",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Health+outcome"
 },
 {
   "source": 16,
   "target": 1,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting"
 },
 {
   "source": 16,
   "target": 2,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Imports"
 },
 {
   "source": 16,
   "target": 3,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Exports"
 },
 {
   "source": 16,
   "target": 4,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Wastage+non-food+use+animal+food"
 },
 {
   "source": 16,
   "target": 5,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 16,
   "target": 6,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods"
 },
 {
   "source": 16,
   "target": 7,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+supply",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing"
 },
 {
   "source": 17,
   "target": 8,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Distribution",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 17,
   "target": 9,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Distribution",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Acquired"
 },
 {
   "source": 19,
   "target": 14,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Utilization",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression"
 },
 {
   "source": 18,
   "target": 11,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumption",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 18,
   "target": 12,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumption",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 18,
   "target": 13,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumption",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health"
 },
 {
   "source": 20,
   "target": 14,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Health+outcome",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression"
 },
 {
   "source": 20,
   "target": 15,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Health+outcome",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes"
 },
 {
   "source": 1,
   "target": 3,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Exports"
 },
 {
   "source": 2,
   "target": 3,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Imports",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Exports"
 },
 {
   "source": 2,
   "target": 4,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Imports",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Wastage+non-food+use+animal+food"
 },
 {
   "source": 2,
   "target": 5,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Imports",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 2,
   "target": 6,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Imports",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods"
 },
 {
   "source": 2,
   "target": 7,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Imports",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing"
 },
 {
   "source": 1,
   "target": 4,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Wastage+non-food+use+animal+food"
 },
 {
   "source": 1,
   "target": 5,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 1,
   "target": 6,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods"
 },
 {
   "source": 1,
   "target": 7,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing"
 },
 {
   "source": 5,
   "target": 6,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods"
 },
 {
   "source": 5,
   "target": 7,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing"
 },
 {
   "source": 6,
   "target": 8,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 7,
   "target": 3,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Exports"
 },
 {
   "source": 7,
   "target": 4,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Wastage+non-food+use+animal+food"
 },
 {
   "source": 7,
   "target": 5,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 7,
   "target": 6,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods"
 },
 {
   "source": 8,
   "target": 7,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+processing"
 },
 {
   "source": 8,
   "target": 9,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Acquired"
 },
 {
   "source": 9,
   "target": 10,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Acquired",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Consumed"
 },
 {
   "source": 10,
   "target": 11,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Consumed",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 11,
   "target": 12,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 12,
   "target": 14,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression"
 },
 {
   "source": 12,
   "target": 15,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes"
 },
 {
   "source": 12,
   "target": 13,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health"
 },
 {
   "source": 13,
   "target": 11,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 13,
   "target": 12,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 13,
   "target": 14,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression"
 },
 {
   "source": 13,
   "target": 15,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes"
 },
 {
   "source": 15,
   "target": 13,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health"
 },
 {
   "source": 14,
   "target": 13,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health"
 },
 {
   "source": 14,
   "target": 15,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes"
 },
 {
   "source": 12,
   "target": 11,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 13,
   "target": 14,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Supplements+and+natural_health",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression"
 },
 {
   "source": 14,
   "target": 15,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Genetic+expression",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes"
 },
 {
   "source": 21,
   "target": 3,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Environmental+and+Agricultural+Influences",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Exports"
 },
 {
   "source": 21,
   "target": 4,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Environmental+and+Agricultural+Influences",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Wastage+non-food+use+animal+food"
 },
 {
   "source": 21,
   "target": 5,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Environmental+and+Agricultural+Influences",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 21,
   "target": 6,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Environmental+and+Agricultural+Influences",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Available+food+supply+Traditional+foods"
 },
 {
   "source": 22,
   "target": 8,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Transport",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 23,
   "target": 8,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Consumption",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Storage"
 },
 {
   "source": 25,
   "target": 9,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Access+and+equity+issues",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Acquired"
 },
 {
   "source": 26,
   "target": 28,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Education+Information+Advertising",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference"
 },
 {
   "source": 26,
   "target": 27,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Education+Information+Advertising",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Knowledge+Attitudes+Beliefs+Behaviours"
 },
 {
   "source": 27,
   "target": 26,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Knowledge+Attitudes+Beliefs+Behaviours",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Education+Information+Advertising"
 },
 {
   "source": 27,
   "target": 28,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Knowledge+Attitudes+Beliefs+Behaviours",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference"
 },
 {
   "source": 28,
   "target": 26,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Education+Information+Advertising"
 },
 {
   "source": 28,
   "target": 27,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Knowledge+Attitudes+Beliefs+Behaviours"
 },
 {
   "source": 28,
   "target": 29,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=income"
 },
 {
   "source": 28,
   "target": 30,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=price"
 },
 {
   "source": 29,
   "target": 30,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=income",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=price"
 },
 {
   "source": 29,
   "target": 28,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=income",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference"
 },
 {
   "source": 30,
   "target": 29,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=price",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=income"
 },
 {
   "source": 30,
   "target": 28,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=price",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference"
 },
 {
   "source": 28,
   "target": 10,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+Consumed"
 },
 {
   "source": 31,
   "target": 11,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Age+sex+physical+status",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 32,
   "target": 12,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Existing+health+status+and+nutrient+requirements",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 34,
   "target": 11,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Contaminants+in+foods",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Nutrient"
 },
 {
   "source": 33,
   "target": 15,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Housing+sanitation",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=health+status+outcomes"
 },
 {
   "source": 28,
   "target": 24,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=food+preference",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumer+demand"
 },
 {
   "source": 24,
   "target": 21,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumer+demand",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Environmental+and+Agricultural+Influences"
 },
 {
   "source": 24,
   "target": 1,
   "value": Math.floor(Math.random() * 100),
   "source_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Consumer+demand",
   "target_url": "https://www150.statcan.gc.ca/n1/en/type/data?text=Food+production+Hunting+and+Harvesting"
 }
]

biHiSankey
  .nodes(exampleNodes)
  .links(exampleLinks)
  .initializeNodes(function (node) {
    node.state = node.parent ? "contained" : "collapsed";
  })
  .layout(LAYOUT_INTERATIONS);

disableUserInterractions(2 * TRANSITION_DURATION);

update();