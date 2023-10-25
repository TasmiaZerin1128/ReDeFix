const settings = require("../settings");
const DOM = require("./DOM");
const path = require('path');
const driver = require("./Driver");
const Rectangle = require("./Rectangle");
const RepairStatistics = require("./RepairStatistics");
const utils = require("./utils");
const fs = require('fs');
const { sendMessage } = require("../socket-connect");

class Failure {
    constructor(webpage, run) {
        this.webpage = webpage;
        this.run = run;
        this.ID = undefined;
        this.range = undefined;
        this.type = 'Unknown-Failure';
        this.checkRepaireLater = false;
        this.repairStats = new RepairStatistics();
        this.ID = utils.getNewFailureID();
        utils.incrementFailureCount();
        this.repairElementHandle = undefined; //Style Element used to inject
        this.repairCSS = undefined; //CSS code for repair
        this.repairCSSComments = undefined; //CSS comments for repair
        this.repairApproach = undefined; //indicated which approach was successfully applied.
        this.removeFailingRepair = true;
        this.repairCombinationResult = [];
        this.durationFailureClassify = undefined;
        this.durationFailureRepair = undefined;
        this.durationWiderRepair = undefined;
        this.durationNarrowerRepair = undefined;
        this.durationWiderConfirmRepair = undefined;
        this.durationNarrowerConfirmRepair = undefined;
    }

    setupHumanStudyData() {
        this.hsData = { //will carry anonymous data only.
            ID: this.ID,
            type: this.type,
            rangeMin: this.range.min,
            rangeMax: this.range.max,
            rectangles: []
        };
        if (this.outputDirectory === undefined)
            throw "\nError - " + this.type + " has no output directory - ID:" + this.ID + "\n";
        this.hsKey = {
            humanStudyDirectory: path.join(this.outputDirectory, 'human-study', 'screenshots'),
            ID: this.ID,
            type: this.type,
            rangeMin: this.range.min,
            rangeMax: this.range.max,
            repairName: [],
            anonymizedImageNames: [],
            viewports: [],
            randomIDsUsed: [],
            rectangles: [],
            scrollY: []
        }
    }

    addRange(range) {
        this.ranges.addRange(range);
    }
    /**
     * Adds a set of ranges to the set of ranges belonging to this failures. Deprecated
     * @param {Ranges} ranges The set of ranges to add to this set of ranges.
     */
    addRanges(ranges) {
        this.ranges.addRanges(ranges);
    }

    getClippingCoordinates(rectangles, extra = 0, maxHeight = 8000) {
        let problemArea = new Rectangle();
        for (let rect of rectangles) {
            if (rect.isMissingValues()) {
                utils.log(this.ID + " " + this.type + " " + this.range.toString());
                utils.log("Warning: cannot use rectangle for cutting screenshot...");
                utils.log(rect.toString(true));
                continue;
            }
            if (rect.height > maxHeight) { //max element height to support.
                utils.log(this.ID + " " + this.type + " " + this.range.toString());
                utils.log("Warning: Human Study - rectangle height exceeds maximum height: " + maxHeight);
                utils.log(rect.toString(true));
                continue;
            }
            if (problemArea.minX === undefined || rect.minX < problemArea.minX)
                problemArea.minX = rect.minX;
            if (problemArea.maxX === undefined || rect.maxX > problemArea.maxX)
                problemArea.maxX = rect.maxX;
            if (problemArea.minY === undefined || Math.max(rect.minY - extra, 0) < problemArea.minY)
                problemArea.minY = Math.max(rect.minY - extra, 0);
            if (problemArea.maxY === undefined || (rect.maxY + extra) > problemArea.maxY)
                problemArea.maxY = (rect.maxY + extra);
        }
        if (problemArea.isMissingValues()) { //use first in the worst case
            utils.log(this.ID + " " + this.type + " " + this.range.toString());
            utils.log("Warning: Human Study - Had to use first rectangle for cutting.");
            let rect = rectangles[0];
            utils.log(rect.toString(true));
            problemArea.minX = rect.minX;
            problemArea.maxX = rect.maxX;
            problemArea.minY = Math.max(rect.minY - extra, 0);
            problemArea.maxY = (rect.maxY + extra);
        }
        problemArea.width = problemArea.maxX - problemArea.minX;
        problemArea.height = problemArea.maxY - problemArea.minY;
        return problemArea;
    }


    async clipScreenshot(rectangles, screenshot, driver, fullViewportWidthClipping, viewport, problemArea = undefined) {
        try {
            if (problemArea === undefined)
                problemArea = this.getClippingCoordinates(rectangles);
            if (problemArea.isMissingValues())
                throw "== Begin Problem Area ==\n" +
                "Missing size for cutting screenshot\n" +
                problemArea + "\n" +
                "== End  Problem  Area ==\n" +
                "==  Begin Rectangles  ==\n" +
                + rectangles + "\n" +
                "==   End  Rectangles  ==\n"
            screenshot = await driver.clipImage(screenshot, problemArea, fullViewportWidthClipping, viewport);
        } catch (passedInErrorMessage) {
            let newErrorMessage = this.ID + ' ' + this.type + ' ' + this.range.toString() + '\n';
            for (let rect of rectangles) {
                newErrorMessage += rect.toString(true) + '\n';
            }
            newErrorMessage += '\n' + passedInErrorMessage;
            throw newErrorMessage;
        }
        return screenshot;
    }

    saveScreenshot(file, screenshot, removeHeader = true) {
        if (removeHeader)
            screenshot = screenshot.split(',')[1];
        fs.writeFile(file, screenshot, 'base64', function (err) {
            if (err) {
                console.log('ERROR IN SAVING IMAGE');
                console.log(err);
            }
        });
    }

    // Get parent width or viewport width and height from wider viewport.
    async getDOMFrom(driver, viewport, pseudoElements = [], root = undefined, xpath = undefined) {
        if (viewport === undefined)
            viewport = this.range.getWider();
        if (driver.currentViewport !== viewport)
            await driver.setViewport(viewport, settings.testingHeight);
        let dom = new DOM(driver, viewport);
        await dom.captureDOM(true, true, pseudoElements, root, xpath);
        return dom;
    }
    
    getDOMStylesCSS(dom) {
        if (this.repairCSS === undefined)
            this.repairCSS = '';
        let traversalStackDOM = [];
        traversalStackDOM.push(dom.root);
        while (traversalStackDOM.length > 0) {
            let domNode = traversalStackDOM.shift();
            let css = this.getRepairCSSFromComputedStyle(domNode.getComputedStyle(), undefined, undefined, false);
            if (css !== '') {
                this.repairCSS +=
                    "   " + domNode.getSelector() + " {\n" +
                    css +
                    "   " + "}\n";
            }
            for (let child of domNode.children) {
                traversalStackDOM.push(child);
            }
        }
        return this.repairCSS;
    }

    getRepairCSSFromComputedStyle(computedStyle, oracleViewport, cushion = 1, scale = true) {
        let css = '';
        for (let property in computedStyle) {
            if (settings.skipCopyingCSSProperties.includes(property))
                continue;
            let widerValues = [];
            let partsWithPX = [];
            if (scale === true &&
                computedStyle[property].includes('px') &&
                !settings.NoScalingCSSProperties.includes(property)) { //Avoid scaling some properties
                let parts = computedStyle[property].split(" ");

                for (let part of parts) {
                    if (part.includes('px')) {
                        let hasComma = false;
                        if (part.includes(',')) {
                            hasComma = true
                            part = part.trim().replace(',', '');
                        }
                        let num = Number(part.trim().replace('px', ''));
                        if (num === undefined || Number.isNaN(num)) {
                            let message = "Error - PX to Number: " + num + "\n" +
                                "Original: " + part + "\n";
                            throw message;
                        }
                        widerValues.push(num);
                        partsWithPX.push(true);
                        if (hasComma) {
                            widerValues.push(',');
                            partsWithPX.push(false);
                        }
                    }
                    else {
                        widerValues.push(part.trim());
                        partsWithPX.push(false);
                    }
                }
            }
            if (widerValues.length > 0 && scale) {
                let values = '';
                for (let i = 0; i < widerValues.length; i++) {
                    let figure = widerValues[i];
                    let withPX = partsWithPX[i];
                    if (withPX === true) {
                        if (figure === 0) {
                            values += '0px ';
                        } else {
                            values += "calc((100vw/" + (oracleViewport + cushion) + ")*" + figure + ") ";
                        }
                    } else {
                        values += figure + ' ';
                    }

                }
                if (values !== '') {
                    css +=
                        "      " + property + ": " + values + "!important; \n";
                }
            }
            else {
                css +=
                    "      " + property + ": " + computedStyle[property] + " !important; \n";
            }
        }
        return css;
    }

    associateRLGNodesToDOMNodes(dom, rlg) {
        let traversalStackDOM = [];
        traversalStackDOM.push(dom.root);
        while (traversalStackDOM.length > 0) {
            let domNode = traversalStackDOM.shift();

            let associatedRLGNode = rlg.getRLGNode(domNode.xpath)
            if (associatedRLGNode !== undefined) {
                domNode.rlgNode = associatedRLGNode;
            } else {
                let associatedXPath = undefined;
                for (let xpath of rlg.map.keys()) {
                    if (domNode.xpath.includes(xpath + '/')) {//xpath is a sub-path of DOM node xpath
                        if (associatedXPath === undefined) {
                            associatedXPath = xpath;
                        } else if (associatedXPath.length <= xpath.length) {
                            associatedXPath = xpath;
                        }
                    }
                }
                if (associatedXPath === undefined)
                    throw "Could not associate RLG node to associate with DOM node xpath: " + domNode.xpath;
                else
                    domNode.rlgNode = rlg.getRLGNode(associatedXPath)
            }

            for (let child of domNode.children) {
                traversalStackDOM.push(child);
            }
        }

    }

    WiderComparisonViewportSpacingCSS(domWider, domMin, widerViewportWidth, pseudoElements) {
        if (this.repairCSS === undefined)
            this.repairCSS = '';
        let traversalStackDOM = [];
        traversalStackDOM.push(domWider.root);
        while (traversalStackDOM.length > 0) {
            let domNode = traversalStackDOM.shift();
            let nvWidthRatio = undefined;
            // let properties = ['min-width', 'width', 'max-width', 'min-height', 'height', 'max-height', 'font-size', 'margin-left', 'margin-right', 'margin-top', 'margin-bottom', 'padding-left', 'padding-right', 'padding-top', 'padding-bottom', 'border-left-width', 'border-right-width', 'border-top-width', 'border-bottom-width']
            let properties = ['min-width', 'width', 'max-width', 'min-height', 'height', 'font-size', 'line-height', 'margin-left', 'margin-right', 'margin-top', 'margin-bottom', 'padding-left', 'padding-right', 'padding-top', 'padding-bottom', 'border-left-width', 'border-right-width', 'border-top-width', 'border-bottom-width']
            let domNodeMin = domMin.getDOMNode(domNode.xpath);
            if (domNodeMin === undefined) {
                assist.log("In Wider not in Min DOM: " + domNode.xpath);
            }
            let computedStylesWider = domNode.getComputedStyle();
            let computedStylesMin = undefined;
            if (domNodeMin !== undefined)
                computedStylesMin = domNodeMin.getComputedStyle();
            //for (let key of properties) {
            //Rescale Ratio
            let widthWider = undefined;
            let widthMin = undefined;
            if (domNode.rectangle !== undefined && domNode.rectangle.width !== undefined && domNode.rectangle.width > 0 && domNode.rectangle.height > 0) {
                widthWider = domNode.rectangle.width;
                domNode.scale = widthWider / widerViewportWidth;
            }
            if (domNodeMin !== undefined && domNodeMin.rectangle !== undefined && domNodeMin.rectangle.width !== undefined && domNodeMin.rectangle.width > 0 && domNodeMin.rectangle.height > 0) {
                widthMin = domNodeMin.rectangle.width;
                domNodeMin.scale = widthMin / this.range.getMinimum();
            }
            if (widthWider !== undefined && widthMin !== undefined)
                domNode.rescale = domNodeMin.scale / domNode.scale;
            //Computed Style of DOM node...
            let css = this.getRepairCSSFromComputedStyle(computedStylesWider, widerViewportWidth);
            if (css !== '') {
                this.repairCSS +=
                    "   " + domNode.getSelector() + " {\n" +
                    css +
                    "   " + "}\n";
            }

            //Computed Style of DOM node pseudoElements...
            for (let pseudoElement of pseudoElements) {
                let computedStylePE = domNode[pseudoElement];
                if (computedStylePE !== undefined) {
                    css = this.getRepairCSSFromComputedStyle(computedStylePE, widerViewportWidth);
                    if (css !== '') {
                        this.repairCSS +=
                            "   " + domNode.getSelector() + "::" + pseudoElement + " {\n" +
                            css +
                            "   " + "}\n";
                    }
                }
            }


            for (let child of domNode.children) {
                traversalStackDOM.push(child);
            }
        }
        //Make sure that all element in DOM min are also in wider.
        traversalStackDOM = [];
        traversalStackDOM.push(domMin.root);
        while (traversalStackDOM.length > 0) {
            let domNode = traversalStackDOM.shift();
            let domNodeWider = domWider.getDOMNode(domNode.xpath);
            if (domNodeWider === undefined) {
                assist.log("In Min not in Wider DOM: " + domNode.xpath)
            }
            for (let child of domNode.children) {
                traversalStackDOM.push(child);
            }
        }
    }

    // Create media rule using range of failure. Including comments.
    makeRuleCSS(css, comments = true, min = this.range.getMinimum(), max = this.range.getMaximum()) {
        if (css === undefined)
            css = this.repairCSS;
        if (css === undefined || css === '')
            return undefined;

        let ruleCSS = '';
        if (comments === true)
            ruleCSS = this.createCSSComment() +
                '/*\n' + this.repairApproach + '\n' + "comparison-wider-classification: " + this.range.widerClassification + '\n*/\n';
        ruleCSS +=
            "@media only screen and (min-width: " + min + "px) and (max-width: " + max + "px){\n" +
            css +
            "}\n";
        return ruleCSS;
    }

    createCSSComment() {
        this.repairCSSComments = "/*\n" +
            "   " + "ID: " + this.ID + " Type: " + this.type + "\n" +
            "   " + "Range: " + this.range.toString() + "\n" +
            "   " + "Node: " + this.node.xpath + "\n";

        if (this.type === utils.FailureType.COLLISION || this.type === utils.FailureType.SMALLRANGE) {
            this.repairCSSComments += "   " + "Sibling: " + this.sibling.xpath + "\n";
            if (this.type === utils.FailureType.COLLISION)
                this.repairCSSComments += "   " + "Parent: " + this.parent.xpath + "\n";
            if (this.type === utils.FailureType.SMALLRANGE) {
                this.repairCSSComments += "   " + "Set: " + this.set + "\n";
                this.repairCSSComments += "   " + "Set Wider: " + this.setWider + "\n";
                this.repairCSSComments += "   " + "Set Narrower: " + this.setNarrower + "\n";

            }

        }
        if (this.type === utils.FailureType.VIEWPORT || this.type === utils.FailureType.PROTRUSION)
            this.repairCSSComments += "   " + "Parent: " + this.parent.xpath + "\n";
        if (this.type === utils.FailureType.WRAPPING)
            for (let rowElement of this.row)
                this.repairCSSComments += "   " + "Row Element: " + rowElement.xpath + "\n";
        this.repairCSSComments += "*/\n"
        return this.repairCSSComments;
    }

    // Using the transform property and scale function with the provided
    // scale value, CSS that targets BODY element is returned.
    getTransformScaleCSS(scaleValue, selector = undefined) {
        let css = "";
        if (selector === undefined)
            css += "   " + "html:nth-of-type(1) {\n";
        else
            css += "   " + selector + " {\n";
        css +=
            "      " + "transform: scale(" + scaleValue + ") !important;\n" +
            "      " + "transform-origin: left top !important;\n" +

            "   " + "}\n";
        return css;
    }

    async resolveRepair(repaired, cssRepairedDirectory, repairName, cssFailedDirectory, css) {
        if (repaired === true) {
            this.repairCombinationResult.push("Repaired");
            this.saveRepairToFile(path.join(cssRepairedDirectory, this.ID + '-' + repairName + ".css"), css);
        } else if (repaired === false) {
            this.saveRepairToFile(path.join(cssFailedDirectory, this.ID + '-' + repairName + ".css"), css);
            await this.deleteRepair();
            css = undefined;
        }
        return css;
    }

    saveRepairToFile(file, css = this.repairCSS) {
        let repairApproachComment = '/*\n' + this.repairApproach + '\n' + "wider-comparison-classification: " + this.range.widerClassification + '\n*/\n';
        if (css !== undefined) {
            fs.appendFileSync(file, this.repairCSSComments + repairApproachComment + css + "\n");
        }
    }

    async setViewportHeightBeforeSnapshot(viewport) {
        if (settings.browserMode === utils.Mode.HEADLESS) {
            let css = undefined;
            if (this.repairElementHandle === undefined) {
                let htmlElement = await driver.getHTMLElement();
                let dom = await this.getDOMFrom(driver, viewport, [], htmlElement);
                css = this.getDOMStylesCSS(dom);
            }

            let pageHeight = await driver.getPageHeightUsingHTMLElement();
            pageHeight = Math.max(pageHeight, settings.testingHeight);
            await driver.setViewport(viewport, pageHeight);
            if (this.repairElementHandle === undefined) {
                this.snapshotCSSElementHandle = await driver.addRepair(css);
            } else {
                await driver.setViewport(viewport, settings.testingHeight);
            }
        }

    }

    async snapshotViewport(driver, viewport, directory, includeClassification = false, clip = true) {
        await this.setViewportHeightBeforeSnapshot(viewport);
        let fullPage = true;
        if (settings.browserMode === utils.Mode.HEADLESS)
            fullPage = false;
        let rectangles = [];
        let elements = [];
        let selectors = this.getSelectors();
        for (let i = 0; i < selectors.length; i++) {
            let currentSelector = selectors[i];
            let element = undefined;
            try {
                element = await driver.getElementBySelector(currentSelector);
                elements.push(element);
            } catch (err) {
                console.log('Error: ' + err);
                elements.push(undefined); 
            }
            if (element != undefined) {
                try {
                    let rectangle = new Rectangle(await driver.getRectangle(element));
                    rectangle.selector = currentSelector;
                    rectangles.push(rectangle);
                } catch (err) {
                    console.log('Error: ' + err);
                    rectangles.push(undefined);
                }
            } else {
                rectangles.push(undefined);
            }
            let screenshot = await driver.screenshot(undefined, fullPage);
            let removeHeader = false;
            if (settings.screenshotHighlights) {
                screenshot = await driver.highlight(rectangles, screenshot);
                removeHeader = true;
            }
            // if (!settings.screenshotFullpage) {
            //     screenshot = await this.clipScreenshot(rectangles, screenshot.split(',')[1], driver, true, viewport);
            // }
            let imageFileName = 'FID-' + this.ID + '-' + this.type.toLowerCase() + '-' + this.range.toString().trim() + '-capture-' + viewport;
            let classification = this.range.getClassificationOfViewport(viewport);
            if (includeClassification && classification !== '-')
                imageFileName += '-' + classification + '.png';
            else
                imageFileName += '.png';
            this.saveScreenshot(path.join(directory, imageFileName), screenshot, removeHeader);
            for (let element of elements) {
                if (element !== undefined)
                    element.dispose();
            }
            await this.resetViewportHeightAfterSnapshots(driver.currentViewport);
        }
    }

    async resetViewportHeightAfterSnapshots(viewport) {
        if (this.repairElementHandle === undefined && this.snapshotCSSElementHandle !== undefined) {
            await driver.removeRepair(this.snapshotCSSElementHandle);
            await this.snapshotCSSElementHandle.dispose();
        }
        if (settings.browserMode === utils.Mode.HEADLESS)
            await driver.setViewport(viewport, settings.testingHeight);
    }

    // Returns an object with pixels child protruding {left, right, top, bottom}.
    calculateProtrusion(parentRect, childRect) {
        let protruding =
        {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0
        }
        if (childRect.minX < parentRect.minX) {
            protruding.left = parentRect.minX - childRect.minX;
        }
        if (childRect.maxX > parentRect.maxX) {
            protruding.right = childRect.maxX - parentRect.maxX;
        }
        if (childRect.minY < parentRect.minY) {
            protruding.top = parentRect.minY - childRect.minY;
        }
        if (childRect.maxY > parentRect.maxY) {
            protruding.bottom = childRect.maxY - parentRect.maxY;
        }
        return protruding;
    }

   /* Returns an object with number of pixels overlapping {left, right, top, bottom}.
    * Zeros for no collision.
    */
    calculateOverlap(nodeRect, siblingRect) {
        let overlap =
        {
            //node to be pushed away on either the x-axis or the y-axis.
            nodeRectToBeCleared: undefined,
            otherNodeRect: undefined,
            xToClear: 0,
            yToClear: 0
        }
        let nodeRectToBeCleared = undefined;
        let otherNodeRect = undefined;
        let xToClear = 0;
        let yToClear = 0;
        if (nodeRect.minX < siblingRect.minX) {
            nodeRectToBeCleared = siblingRect;
            otherNodeRect = nodeRect;
        } else if (nodeRect.minX > siblingRect.minX) {
            nodeRectToBeCleared = nodeRect;
            otherNodeRect = siblingRect;
        } else if (nodeRect.minX === siblingRect.minX) {
            if (nodeRect.minY < siblingRect.minY) {
                nodeRectToBeCleared = siblingRect;
                otherNodeRect = nodeRect;
            } else if (nodeRect.minY > siblingRect.minY) {
                nodeRectToBeCleared = nodeRect;
                otherNodeRect = siblingRect;
            } else if (nodeRect.minY === siblingRect.minY) {
                if (nodeRect.maxX < siblingRect.maxX) {
                    nodeRectToBeCleared = siblingRect;
                    otherNodeRect = nodeRect;
                } else if (nodeRect.maxX > siblingRect.maxX) {
                    nodeRectToBeCleared = nodeRect;
                    otherNodeRect = siblingRect;
                } else if (nodeRect.maxX === siblingRect.maxX) {
                    if (nodeRect.maxY < siblingRect.maxY) {
                        nodeRectToBeCleared = siblingRect;
                        otherNodeRect = nodeRect;
                    } else if (nodeRect.maxY > siblingRect.maxY) {
                        nodeRectToBeCleared = nodeRect;
                        otherNodeRect = siblingRect;
                    } else if (nodeRect.maxY === siblingRect.maxY) {
                        /**
                         * If they are equal rectangles break the tie by xpath length.
                         */
                        let sortedByXpath = [nodeRect.xpath, siblingRect.xpath].sort();
                        if (sortedByXpath[0] === nodeRect.xpath) {
                            nodeRectToBeCleared = siblingRect;
                            otherNodeRect = nodeRect;
                        } else {
                            nodeRectToBeCleared = nodeRect;
                            otherNodeRect = siblingRect;
                        }

                    }
                }
            }
        }
        if (utils.areOverlapping(nodeRectToBeCleared, otherNodeRect)) {
            xToClear = otherNodeRect.maxX - nodeRectToBeCleared.minX + 1;
            yToClear = otherNodeRect.maxY - nodeRectToBeCleared.minY + 1;
        } else {
            xToClear = 0;
            yToClear = 0;
        }

        overlap.nodeRectToBeCleared = nodeRectToBeCleared;
        overlap.otherNodeRect = otherNodeRect;
        overlap.xToClear = xToClear;
        overlap.yToClear = yToClear;
        return overlap;
    }

    // DOM level verification of the reported failures
    async classify(driver, classificationFile, snapshotDirectory, bar) {
        this.durationFailureClassify = new Date();
        let range = this.range;
        let counter = 0;

        await driver.setViewport(range.getNarrower(), settings.testingHeight);
        range.narrowerClassification = await this.isFailing(driver, range.getNarrower(), classificationFile, range) ? 'TP' : 'FP';
        if (settings.screenshotNarrower === true)
            await this.snapshotViewport(driver, range.getNarrower(), snapshotDirectory, true);

        await driver.setViewport(range.getMinimum(), settings.testingHeight);

        range.minClassification = await this.isFailing(driver, range.getMinimum(), classificationFile, range) ? 'TP' : 'FP';
        if (settings.screenshotMin === true)
            await this.snapshotViewport(driver, range.getMinimum(), snapshotDirectory, true);
        // if (settings.humanStudy === true)
        //     await this.screenshotForHumanStudy('Failure');

        await driver.setViewport(range.getMiddle(), settings.testingHeight);
        range.midClassification = await this.isFailing(driver, range.getMiddle(), classificationFile, range) ? 'TP' : 'FP';
        if (settings.screenshotMid === true)
            await this.snapshotViewport(driver, range.getMiddle(), snapshotDirectory, true);

        await driver.setViewport(range.getMaximum(), settings.testingHeight);

        range.maxClassification = await this.isFailing(driver, range.getMaximum(), classificationFile, range) ? 'TP' : 'FP';
        if (settings.screenshotMax === true)
            await this.snapshotViewport(driver, range.getMaximum(), snapshotDirectory, true);

        await driver.setViewport(range.getWider(), settings.testingHeight);

        range.widerClassification = await this.isFailing(driver, range.getWider(), classificationFile, range) ? 'TP' : 'FP';
        if (settings.screenshotWider === true)
            await this.snapshotViewport(driver, range.getWider(), snapshotDirectory, true);

        counter++;
        bar.tick();
        sendMessage("Classify", {'counter': counter, 'total': utils.failureCount});
        this.durationFailureClassify = new Date() - this.durationFailureClassify;
    }

    // Repair this failure
    async repair(driver, directory, bar, webpage, run) {
        this.durationFailureRepair = new Date();
        await this.findRepair(driver, directory, bar, webpage, run);
        this.durationFailureRepair = new Date() - this.durationFailureRepair;
    }

    printWorkingRepairs(file, webpage, run) {
        for (let i = 0; i < this.repairCombinationResult.length; i++) {
            let repairArray = settings.repairCombination[i];
            let repairName = '';
            for (let subRepair of repairArray) {
                if (repairName === '')
                    repairName = subRepair;
                else
                    repairName += "-" + subRepair;
            }
            let xpaths = ''
            if (this.type === utils.FailureType.COLLISION || this.type === utils.FailureType.SMALLRANGE)
                xpaths = this.node.xpath + ',' + this.sibling.xpath;
            else if (this.type === utils.FailureType.VIEWPORT || this.type === utils.FailureType.PROTRUSION)
                xpaths = this.node.xpath + ',' + this.parent.xpath;
            else if (this.type === utils.FailureType.WRAPPING)
                xpaths = this.node.xpath + ',' + this.row[0].xpath;
            let repairOutcome = this.repairCombinationResult[i];
            let text =
                EOL + webpage + "," + run + "," + this.ID + "," + this.type + "," + this.range.getMinimum() + "," + this.range.getMaximum() + "," + xpaths + "," + this.range.narrowerClassification + "," + this.range.minClassification + "," + this.range.midClassification + "," + this.range.maxClassification + "," + this.range.widerClassification + "," + repairName + "," + repairOutcome;
            fs.appendFileSync(file, text, function (err) {
                if (err) throw err;
            });
        }
    
        if (this.repairCombinationResult.length === 0) {
            let xpaths = ''
            if (this.type === utils.FailureType.COLLISION || this.type === utils.FailureType.SMALLRANGE)
                xpaths = this.node.xpath + ',' + this.sibling.xpath
            else if (this.type === utils.FailureType.VIEWPORT || this.type === utils.FailureType.PROTRUSION)
                xpaths = this.node.xpath + ',' + this.parent.xpath
            else if (this.type === utils.FailureType.WRAPPING)
                xpaths = this.node.xpath + ',' + this.row[0].xpath;
            let repairOutcome = "Skipped";
            let repairName = 'None';
            let text =
                EOL + webpage + "," + run + "," + this.ID + "," + this.type + "," + this.range.getMinimum() + "," + this.range.getMaximum() + "," + xpaths + "," + this.range.narrowerClassification + "," + this.range.minClassification + "," + this.range.midClassification + "," + this.range.maxClassification + "," + this.range.widerClassification + "," + repairName + "," + repairOutcome;
            fs.appendFileSync(file, text, function (err) {
                if (err) throw err;
            });
        }
    }

    async findRepair(driver, outputDirectory, bar, webpage, run) {
        let cssRepairedDirectory = path.join(outputDirectory, 'CSS', 'repaired');
        let cssFailedDirectory = path.join(outputDirectory, 'CSS', 'failed');


        let failureViewport = this.range.getMinimum();
        await driver.setViewport(failureViewport, settings.testingHeight);
        if (this.range.minClassification === 'TP') { //Needs Repair.
            //if (await this.isFailing(driver)) { //Attempt to repair failure only if there is a problem.
            //if (await this.isFailing(driver)) { //Attempt to repair failure only if there is a problem.
                this.repairCombinationResult = [];

                //let repairBar = new ProgressBar('Repair ' + this.ID + '      [:bar] :etas Attempt:       :token1/' + settings.repairCombination.length, { incomplete: ' ', total: settings.repairCombination.length, width: 25 })
                let count = 0;
                for (let repair of settings.repairCombination) {
                    this.repairApproach = '';
                    let css = undefined;
                    let repaired = false;
                    let attemptedRepair = false;
                    let rlgNode = this.node.getParentAtViewport(this.range.getWider());
                    let pseudoElements = []//['after', 'before']//, 'first-letter', 'first-line'];
                    let repairName = ''

                    if (repair.includes('Transform-Wider')) {
                        this.durationWiderRepair = new Date();
                        if (this.range.widerClassification === 'FP') {
                            attemptedRepair = true;
                            repairName = repair[0];
                            await driver.setViewport(this.range.getWider(), settings.testingHeight);
                            let htmlElement = await driver.getHTMLElement();
                            let domWider = await this.getDOMFrom(driver, this.range.getWider(), pseudoElements, htmlElement);
                            css = this.getDOMStylesCSS(domWider);
                            if (!(settings.humanStudy && settings.humanStudyMoreViewportWidth && settings.browserMode === assist.Mode.HEADLESS))
                                css = this.makeRuleCSS(css);
                            //Scale ratio specifically made for viewport...
                            let min = this.range.getMinimum();
                            let max = this.range.getMaximum();
                            if (!(settings.humanStudy && settings.humanStudyMoreViewportWidth && settings.browserMode === assist.Mode.HEADLESS))
                            {
                                min = settings.testingWidthMin;
                                max= settings.testingRangeMax;
                            }
                            for (let failureViewport = min; failureViewport <= max; failureViewport++) {
                                let scaleValue = failureViewport / this.range.getWider();
                                let scaleCSS = this.getTransformScaleCSS(scaleValue);
                                let scaleCSSRule = this.makeRuleCSS(scaleCSS, false, failureViewport, failureViewport);
                                css += scaleCSSRule;
                            }
                            this.durationWiderConfirmRepair = new Date();
                            repaired = await this.isRepaired(css, repairName, outputDirectory, webpage, run);
                            this.durationWiderConfirmRepair = new Date() - this.durationWiderConfirmRepair;
                            css = await this.resolveRepair(repaired, cssRepairedDirectory, repairName, cssFailedDirectory, css);
                        }
                        this.durationWiderRepair = new Date() - this.durationWiderRepair;
                    }
                    if (repair.includes('Transform-Wider-Targeted')) {
                        if (this.range.widerClassification === 'FP') {
                            let parent = this.node.getParentAtViewport(this.range.getWider());
                            attemptedRepair = true;
                            repairName = repair[0];
                            await driver.setViewport(this.range.getWider(), settings.testingHeight);
                            let root = await driver.getElementBySelector(parent.getSelector());
                            let domWider = await this.getDOMFrom(driver, this.range.getWider(), pseudoElements, root, parent.xpath);
                            css = this.getDOMStylesCSS(domWider);
                            css = this.makeRuleCSS(css);
                            //Scale ratio specifically made for viewport...
                            for (let failureViewport = this.range.getMinimum(); failureViewport <= this.range.getMaximum(); failureViewport++) {
                                let scaleValue = failureViewport / this.range.getWider();
                                let scaleCSS = this.getTransformScaleCSS(scaleValue, parent.getSelector());
                                let scaleCSSRule = this.makeRuleCSS(scaleCSS, false, failureViewport, failureViewport);
                                css += scaleCSSRule;
                            }
                            repaired = await this.isRepaired(css, repairName, outputDirectory, webpage, run);
                            css = await this.resolveRepair(repaired, cssRepairedDirectory, repairName, cssFailedDirectory, css);
                        }
                    }
                    if (repair.includes('Transform-Narrower')) {
                        this.durationNarrowerRepair = new Date();
                        let narrowerViewportWidth = this.range.getNarrower();
                        if (narrowerViewportWidth >= settings.testingWidthMin && this.range.narrowerClassification === 'FP') {
                            attemptedRepair = true;
                            repairName = repair[0];
                            await driver.setViewport(this.range.getNarrower(), settings.testingHeight);
                            let htmlElement = await driver.getHTMLElement();
                            let domNarrower = await this.getDOMFrom(driver, this.range.getNarrower(), pseudoElements, htmlElement);
                            css = this.getDOMStylesCSS(domNarrower);
                            if (!(settings.humanStudy && settings.humanStudyMoreViewportWidth && settings.browserMode === assist.Mode.HEADLESS))
                                css = this.makeRuleCSS(css);
                            //Scale ratio specifically made for viewport...
                            let min = this.range.getMinimum();
                            let max = this.range.getMaximum();
                            if (!(settings.humanStudy && settings.humanStudyMoreViewportWidth && settings.browserMode === assist.Mode.HEADLESS))
                            {
                                min = settings.testingWidthMin;
                                max= settings.testingRangeMax;
                            }
                            for (let failureViewport = min; failureViewport <= max; failureViewport++) {
                                let scaleValue = failureViewport / this.range.getNarrower();
                                let scaleCSS = this.getTransformScaleCSS(scaleValue);
                                let scaleCSSRule = this.makeRuleCSS(scaleCSS, false, failureViewport, failureViewport);
                                css += scaleCSSRule;
                            }
                            this.durationNarrowerConfirmRepair = new Date();
                            repaired = await this.isRepaired(css, repairName, outputDirectory, webpage, run);
                            this.durationNarrowerConfirmRepair = new Date() - this.durationNarrowerConfirmRepair;
                            css = await this.resolveRepair(repaired, cssRepairedDirectory, repairName, cssFailedDirectory, css);
                        }
                        this.durationNarrowerRepair = new Date() - this.durationNarrowerRepair;
                    }

                    if (repair.includes('widerComparisonViewportSpacing')) {
                        if (this.range.widerClassification === 'FP') {
                            attemptedRepair = true;
                            repairName = repair[0];
                            let domMin = await this.getDOMFrom(driver, this.range.getMinimum(), pseudoElements);
                            let widerViewportWidth = this.range.getWider();
                            let domWider = await this.getDOMFrom(driver, widerViewportWidth, pseudoElements);
                            this.WiderComparisonViewportSpacingCSS(domWider, domMin, widerViewportWidth, pseudoElements)
                            css = this.makeRuleCSS();
                            repaired = await this.isRepaired(css, repairName, outputDirectory, webpage, run);
                            css = await this.resolveRepair(repaired, cssRepairedDirectory, repairName, cssFailedDirectory, css);
                        }
                    }
                    if (repair.includes('narrowerComparisonViewportSpacing')) {
                        repairName = repair[0];
                        let narrowerViewportWidth = this.range.getNarrower();
                        if (narrowerViewportWidth >= settings.testingWidthMin && this.range.narrowerClassification === 'FP') {
                            attemptedRepair = true;
                            let domMin = await this.getDOMFrom(driver, this.range.getMinimum(), pseudoElements);
                            let domNarrower = await this.getDOMFrom(driver, narrowerViewportWidth, pseudoElements);
                            this.narrowerComparisonViewportSpacingCSS(domNarrower, domMin, narrowerViewportWidth, pseudoElements);
                            css = this.makeRuleCSS();
                            repaired = await this.isRepaired(css, repairName, outputDirectory, webpage, run);
                            css = await this.resolveRepair(repaired, cssRepairedDirectory, repairName, cssFailedDirectory, css);
                        }
                    }

                    if (repair.includes('widerComparisonElements')) {
                        if (this.range.widerClassification === 'FP') {
                            attemptedRepair = true;
                            let domWider = await this.getDOMFrom(driver, this.range.getWider(), pseudoElements);
                            this.associateRLGNodesToDOMNodes(domWider, this.node.rlg);
                            let ancestorLevel = 0;
                            while (repaired === false && rlgNode !== undefined) {
                                ancestorLevel++;
                                if (this.ID == 2)
                                    assist.log('Level: ' + ancestorLevel);
                                repairName = repair + '-' + ancestorLevel;
                                this.widerComparisonElements(domWider, rlgNode);
                                css = this.makeRuleCSS();
                                repaired = await this.isRepaired(css, repairName, outputDirectory, webpage, run);
                                if (repaired === false) {
                                    this.saveRepairToFile(path.join(cssFailedDirectory, this.ID + '-' + repairName + ".css"), css)
                                    await this.deleteRepair();
                                    css = undefined;
                                    rlgNode = rlgNode.getParentAtViewport(this.range.getWider());
                                }
                                else if (repaired === true) {
                                    this.repairCombinationResult.push("Repaired");
                                    this.saveRepairToFile(path.join(cssRepairedDirectory, this.ID + '-' + repairName + ".css"), css)
                                }
                            }
                        }
                    }
                    if (attemptedRepair) {
                        if (repaired) {
                            // this.repairCombinationResult.push("Repaired");
                            // this.saveRepairToFile(path.join(cssRepairedDirectory, this.repairApproach + ".css"))
                            if (settings.aftermath) {
                                //look at aftermath (Second RLG extraction)
                                let aftermathDirectory = path.join(outputDirectory, 'repair-aftermath', this.ID.toString(), repairName);
                                fs.mkdirSync(aftermathDirectory, { recursive: true });
                                let aftermathDOMsDirectory = path.join(aftermathDirectory, "DOMs");
                                fs.mkdirSync(aftermathDOMsDirectory);
                                let aftermathFile = path.join(aftermathDirectory, 'Failures.csv');
                                let rlg = await this.getNewRLG(settings.testingWidthMin, settings.testingWidthMax, aftermathDOMsDirectory);
                                await rlg.classifyFailures(driver, aftermathDirectory + path.sep + 'Classifications.txt', aftermathDirectory);
                                rlg.printFailuresCSV(aftermathFile, webpage, run, repair, this.ID);
                                rlg.printGraph(path.join(aftermathDirectory, 'RLG.txt'));
                            }
                        }
                        else {
                            this.repairCombinationResult.push("Failed");
                            // this.saveRepairToFile(path.join(cssFailedDirectory, this.repairApproach + ".css"))
                        }
                    }
                    else {
                        this.repairCombinationResult.push("Not Applicable");
                    }
                    if (repaired) { //Repair worked
                        this.repairStats.repairs++;
                        this.checkRepairLater = false;
                        //bar.tick();
                    } else {//Repair did not work
                        this.checkRepairLater = true;
                        if (css !== undefined) {
                            //assist.log(css);
                            await driver.removeRepair(this.repairElementHandle);
                            this.repairCSS = undefined;
                            await this.repairElementHandle.dispose();
                            css = undefined;
                        }
                    }
                    //delete later this is to always remove the repair so it does not have an effect on others.
                    if (css !== undefined) {
                        //assist.log(css);
                        await driver.removeRepair(this.repairElementHandle);
                        this.repairCSS = undefined;
                        await this.repairElementHandle.dispose();
                    }
                    count++;
                    //repairBar.tick({ 'token1': count });
                    if (bar.total - bar.curr === 1)
                        bar.tick({ 'token1': "Completed" });
                    else
                        bar.tick({ 'token1': "FID:" + this.ID });

                }
        } else { //FP 
            this.repairStats.doesNotNeedRepair++;
            if (bar.total - bar.curr === 1)
                bar.tick(settings.repairCombination.length, { 'token1': "Completed" });
            else
                bar.tick(settings.repairCombination.length, { 'token1': "FID:" + this.ID });
        }

    }

       
        // Delete repair 
    async deleteRepair() {
        if (this.repairElementHandle !== undefined) {
            await driver.removeRepair(this.repairElementHandle);
            this.repairCSS = undefined;
            await this.repairElementHandle.dispose();
        }
    }
        
        // Inject passed in CSS as a repair and check if repaired.
        async isRepaired(css, repairName, outputDirectory, webpage, run) {
            let repaired = false;
            if (css !== undefined) {
                let miniRLGDirectory = path.join(outputDirectory, 'mini-rlg', this.ID.toString(), repairName);
                fs.mkdirSync(miniRLGDirectory, { recursive: true });
                await this.injectRepair(css);
                if (settings.repairConfirmUsing === RepairConfirmed.RLG)
                    repaired = await this.isRepairedRLG(driver, outputDirectory, miniRLGDirectory, webpage, run, repairName, this.ID);
                else if (settings.repairConfirmUsing === RepairConfirmed.DOM)
                    repaired = await this.isRepairedDOM(driver, repairName, outputDirectory);
                else if (settings.repairConfirmUsing === RepairConfirmed.DOMRLG) {
                    repaired = await this.isRepairedDOM(driver, repairName, outputDirectory);
                    if (repaired)
                        repaired = await this.isRepairedRLG(driver, outputDirectory, miniRLGDirectory, webpage, run, repairName, this.ID);
                }
            }
            if (settings.humanStudy === true && repaired)
                await this.screenshotForHumanStudy(repairName);
            return repaired;
        }
        
        // Injects CSS into the page then applies the delay in global settings
        async injectRepair(css) {
            this.repairElementHandle = await driver.addRepair(css);
        }

        // Checks if isFailing after the delay is executed without using an RLG.
        async isRepairedDOM(driver, repairName, directory) {
            let failureViewport = this.range.getMinimum();
            if (driver.currentViewport !== failureViewport)
                await driver.setViewport(failureViewport, settings.testingHeight);
            let failed = await this.isFailing(driver)
            let repaired = !failed;
            if (repaired && directory !== undefined) {
                let imageFileName = 'FID-' + this.ID + '-' + this.type.toLowerCase() + '-' + this.range.toShortString().trim() + '-capture-' + failureViewport + '-repaired-' + repairName + '.png';
                let imagePath = path.join(directory, "snapshots", imageFileName);
                await this.screenshot(driver, imagePath, settings.screenshotHighlights);
                let maxViewport = this.range.getMaximum();
                if (driver.currentViewport !== maxViewport)
                    await driver.setViewport(maxViewport, settings.testingHeight);
                imageFileName = 'FID-' + this.ID + '-' + this.type.toLowerCase() + '-' + this.range.toShortString().trim() + '-capture-' + maxViewport + '-repaired-' + repairName + '.png';
                imagePath = path.join(directory, "snapshots", imageFileName);
                await this.screenshot(driver, imagePath, settings.screenshotHighlights);
            } else if (!repaired && directory !== undefined) {
                if (settings.screenshotFailingRepairs) {
                    let imageFileName = 'FID-' + this.ID + '-' + this.type.toLowerCase() + '-' + this.range.toShortString().trim() + '-capture-' + failureViewport + '-FAILED-DOM-' + repairName + '.png';
                    let imagePath = path.join(directory, "snapshots", imageFileName);
                    await this.screenshot(driver, imagePath, settings.screenshotHighlights);
                }
            }
            return repaired;
        }
    
        async isRepairedRLG(driver, directory, miniRLGDirectory, webpage, run, repair, id) {
            let dir = directory;
            if (miniRLGDirectory !== undefined)
                dir = miniRLGDirectory;
            let rlg = await this.getNewRLG(undefined, undefined, dir);
            //rlg.printGraph(directory + path.sep + this.ID + '-RLG-' + this.repairApproach + '.txt');
            let failureExists = rlg.hasFailure(this);
            if (failureExists === false) {
                if (miniRLGDirectory !== undefined && webpage !== undefined && run !== undefined && repair !== undefined && id !== undefined) {
                    rlg.printFailuresCSV(path.join(miniRLGDirectory, 'mini-failures.csv'), webpage, run, repair, id);
                }
                let failureViewport = this.range.getMinimum();
                if (driver.currentViewport !== failureViewport)
                    await driver.setViewport(failureViewport, settings.testingHeight);
                let imageFileName = 'FID-' + this.ID + '-' + this.type.toLowerCase() + '-' + this.range.toShortString().trim() + '-capture-' + failureViewport + '-repaired-' + repair + '.png';
                let imagePath = path.join(directory, "snapshots", imageFileName);
                await this.screenshot(driver, imagePath, settings.screenshotHighlights);
                // Take image of repair at maximum of viewport range... 
                let maxViewport = this.range.getMaximum();
                if (driver.currentViewport !== maxViewport)
                    await driver.setViewport(maxViewport, settings.testingHeight);
                imageFileName = 'FID-' + this.ID + '-' + this.type.toLowerCase() + '-' + this.range.toShortString().trim() + '-capture-' + maxViewport + '-repaired-' + repair + '.png';
                imagePath = path.join(directory, "snapshots", imageFileName);
                await this.screenshot(driver, imagePath);
                return true;
            } else {
                if (settings.screenshotFailingRepairs) {
                    let failureViewport = this.range.getMinimum();
                    if (driver.currentViewport !== failureViewport)
                        await driver.setViewport(failureViewport, settings.testingHeight);
                    let imageFileName = 'FID-' + this.ID + '-' + this.type.toLowerCase() + '-' + this.range.toShortString().trim() + '-capture-' + failureViewport + '-FAILED-RLG-' + repair + '.png';
                    let imagePath = path.join(directory, "snapshots", imageFileName);
                    await this.screenshot(driver, imagePath, settings.screenshotHighlights);
                }
    
                return false;
            }
        }
}

module.exports = Failure;