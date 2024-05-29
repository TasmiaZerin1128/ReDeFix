const Failure = require('./Failure');
const settings = require('../settings.js');
const utils = require('./utils.js');
const Rectangle = require('./Rectangle.js');

class ProtrusionFailure extends Failure {
    constructor(node, parent, range, newParent, outputDirectory, webpage, run) {
        super(webpage, run);
        this.node = node;
        this.parent = parent;
        this.newParent = newParent;
        this.range = range;
        this.type = utils.FailureType.PROTRUSION;
        this.outputDirectory = outputDirectory;
        this.protudingArea = null;
        this.images = [];
    }

    equals(otherFailure) {
        if (this.type !== otherFailure.type)
            return false;
        return this.node.xpath === otherFailure.node.xpath && this.parent.xpath === otherFailure.parent.xpath;
    }

    // Check to see if the passed in failure resembles this failure.
    isEquivalent(otherFailure) {
        if (otherFailure.type === utils.FailureType.VIEWPORT) {
            return false;
        } else if (otherFailure.type === utils.FailureType.PROTRUSION) {
            if (this.node.xpath === otherFailure.node.xpath)
                return true;
            let rlgDescendants = otherFailure.node.getDescendantRLGNodesAtViewport(otherFailure.range.getWider());
            for (let descendant of rlgDescendants) {
                if (descendant.xpath === this.node.xpath)//Protruding node in mini-RLG is a descendant of the full-RLG(wider) protruding element.
                    return true;
            }
        } else if (otherFailure.type === utils.FailureType.COLLISION) {
            return (this.node.xpath === otherFailure.node.xpath || this.node.xpath === otherFailure.sibling.xpath)
        } else {
            return undefined;
        }
    }

    // Get xpaths of this failure
    getXPaths() {
        let xpaths = [];
        xpaths.push(this.node.xpath);
        xpaths.push(this.parent.xpath);
        return xpaths;
    }

    getSelectors() {
        let selectors = [];
        selectors.push(this.node.getSelector());
        selectors.push(this.parent.getSelector());
        return selectors;
    }
    print(file) {
        let text = '|  |--[ ' + this.type + ' (' + this.ID + '): ' + this.range.toString() + ' ]';
        utils.printToFile(file, text);
        text = '|  |  |--[ Parent: ' + this.parent.xpath + ' ]';
        utils.printToFile(file, text);
    }
    printClassified(file) {
        let text = '|  |--[ ' + this.type + ' (' + this.ID + '): ' + this.range.toClassifiedString() + ' ]';
        utils.printToFile(file, text);
        text = '|  |  |--[ Parent: ' + this.parent.xpath + ' ]';
        utils.printToFile(file, text);
    }

    // Return true if the child is protruding from the parent
    async isFailing(driver, viewport, file, range) {
        let child = await driver.getElementBySelector(this.node.getSelector());
        let parent = await driver.getElementBySelector(this.parent.getSelector());
        let childRect = new Rectangle(await driver.getRectangle(child));
        let parentRect = new Rectangle(await driver.getRectangle(parent));
        if (parentRect.visible === false || parentRect.validSize === false || parentRect.positiveCoordinates == false)
            return false;
        if (childRect.visible === false || childRect.validSize === false || childRect.positiveCoordinates == false)
            return false;
        let newParentRect = undefined;
        let protruding = this.calculateProtrusion(parentRect, childRect);
        let tol = 0;
        if (settings.tolerance.protrusion !== undefined && settings.tolerance.protrusion > 0)
            tol = settings.tolerance.protrusion;
        let result = (protruding.top > tol || protruding.bottom > tol || protruding.right > tol || protruding.left > tol);
        if (file !== undefined) {
            try {
                if (this.newParent.xpath !== undefined && this.node.xpath !== this.newParent.xpath) {
                    let newParent = await driver.getElementBySelector(this.newParent.getSelector());
                    newParentRect = new Rectangle(await driver.getRectangle(newParent));
                }
            }
            catch (e) {
                //no need to report missing new parent.
            }
            let classification = result ? 'TP' : 'FP';
            let text = 'ID: ' + this.ID + ' Type: ' + this.type + ' Range:' + range.toString() + ' Viewport:' + viewport + ' Classification: ' + classification;
            utils.printToFile(file, text);
            text = '|  |--[ Left-P: ' + protruding.left + ' Right-P: ' + protruding.right + ' Top-P: ' + protruding.top + ' Bottom-P: ' + protruding.bottom + ' ]';
            utils.printToFile(file, text);
            text = '|--[ Parent: ' + this.parent.xpath + ' ]';
            utils.printToFile(file, text);
            text = '|  |--[ minX: ' + parentRect.minX + ' maxX: ' + parentRect.maxX + ' minY: ' + parentRect.minY + ' maxY: ' + parentRect.maxY + ' width: ' + parentRect.width + ' height: ' + parentRect.height + ' ]';
            utils.printToFile(file, text);
            text = '|--[ Child: ' + this.node.xpath + ' ]';
            utils.printToFile(file, text);
            text = '|  |--[ minX: ' + childRect.minX + ' maxX: ' + childRect.maxX + ' minY: ' + childRect.minY + ' maxY: ' + childRect.maxY + ' width: ' + childRect.width + ' height: ' + childRect.height + ' ]';
            utils.printToFile(file, text);

            if (newParentRect !== undefined) {
                text = '|--[ New Parent: ' + this.newParent.xpath + ' ]';
                utils.printToFile(file, text);
                text = '|  |--[ minX: ' + newParentRect.minX + ' maxX: ' + newParentRect.maxX + ' minY: ' + newParentRect.minY + ' maxY: ' + newParentRect.maxY + ' width: ' + newParentRect.width + ' height: ' + newParentRect.height + ' ]';
                utils.printToFile(file, text);
            }
        }
        return result;
    }

    async findAreasOfConcern() {
        if (this.protudingArea.top == 0 && this.protudingArea.bottom == 0 && this.protudingArea.left == 0 && this.protudingArea.right == 0) {
            console.log('False Positive case of Element Protrusion');
            return false;
        }
        return true;
    }

    async isObservable(driver, viewport, file, snapshotDirectory, range) {
        let xpaths = this.getXPaths();
        if (xpaths[0] === xpaths[1]) {
            console.log("Something went wrong, program went to compare the same element to self");
            return;
        } 

        let child = await driver.getElementBySelector(this.node.getSelector());
        let parent = await driver.getElementBySelector(this.parent.getSelector());
        let childRect = new Rectangle(await driver.getRectangle(child));
        let parentRect = new Rectangle(await driver.getRectangle(parent));

        this.protudingArea = this.calculateProtrusion(parentRect, childRect);

        let aoc = await this.findAreasOfConcern();
        if(aoc) {
            await this.takeImages(child, parent, childRect, parentRect, driver, viewport, snapshotDirectory);
            return true;
        }
        return false;
    }

    async takeImages(child, parent, childRect, parentRect, driver, viewport, snapshotDirectory) {

        let opacityChild = await driver.getOpacity(child);
        
        let opacityParent = await driver.getOpacity(parent);

        driver.scroll(child);
        // this.firstImageScrollOffsetX = await driver.getPageScrollWidth();
        // this.firstImageScrollOffsetY = await driver.getPageScrollHeight();

        await driver.setViewport(viewport, settings.testingHeight);

        //Take a screenshot with both elements hidden
        await driver.setOpacity(child, 0);
        await driver.setOpacity(parent, 0);

        let imagePath = viewport + '-imgNoElemets';
        await this.screenshotViewportforVerification(driver, viewport, imagePath, snapshotDirectory, true);
        console.log("Took image for no elements!!!!!!!!!!");

        // Take a screenshot with only the back element visible
        await driver.setOpacity(parent, opacityParent);
        await driver.page.waitForTimeout(100);
        imagePath = viewport + '-imgBack';
        await this.screenshotViewportforVerification(driver, viewport, imagePath, snapshotDirectory, true);
        console.log("Took an image of back!!!!!!!!!!");


        await driver.setOpacity(child, opacityChild);
        await driver.page.waitForTimeout(100);
        imagePath = viewport + '-imgFront';
        await this.screenshotViewportforVerification(driver, viewport, imagePath, snapshotDirectory, true);
        console.log("Took an image of front!!!!!!!!!!");
    }

}

module.exports = ProtrusionFailure;