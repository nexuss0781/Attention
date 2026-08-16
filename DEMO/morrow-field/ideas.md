# Nexuss-auth Dashboard — Design Directions

## Three approaches

### 1. Monochrome Control Plane
**Very Brief Intro:** A precise black-and-white operations console that makes identity infrastructure feel calm, legible, and trustworthy rather than bureaucratic. Dense utility is balanced with generous whitespace and tactile surface contrast.
**Probability:** 0.07

### 2. Paper Credential Studio
**Very Brief Intro:** A bright editorial workspace inspired by archival documents, stamped credentials, and carefully organized technical manuals. It would make project setup feel deliberate and human.
**Probability:** 0.03

### 3. Signal Workshop
**Very Brief Intro:** A warm dark-room interface that treats authorization methods as live signal paths. The mood is more expressive and experimental, with luminous indicators used sparingly.
**Probability:** 0.09

## Chosen approach — Monochrome Control Plane

### Design Movement
Swiss International Style meets contemporary developer infrastructure tooling: ordered, typographic, functional, and quietly technical.

### Core Principles
1. Prioritize decisive, structured information: the project is the central object and settings remain scannable.
2. Use contrast, spacing, and type hierarchy instead of decorative color for emphasis.
3. Make default paths feel effortless while advanced configuration remains available but visually contained.
4. Treat machine-readable information as a first-class visual object so humans and AI agents share the same workflow.

### Color Philosophy
Pure near-black establishes authority and focus; warm white softens long working sessions; a single electric white-on-black state signals the active, trusted system. Grey is used only for secondary infrastructure details, never as a substitute for hierarchy.

### Layout Paradigm
An offset operations rail runs down the left while the workbench has a broad, asymmetric content field. Project context remains anchored in a slim top utility bar; the main canvas uses stacked “workflow sheets” rather than a symmetric card grid.

### Signature Elements
1. The three-orbit Nexuss-auth mark appears as a small black disc with its white symbol.
2. Thin registration lines and compact monospace labels frame project identifiers and code snippets.
3. Provider toggles resemble hardware controls: a clear label, state light, and explicit default/advanced annotation.

### Interaction Philosophy
The primary creation path is direct and low-friction. Advanced panels disclose on demand; every setting clarifies its resulting SDK/CLI effect. Buttons feel physical through subtle press feedback and toggles provide immediate textual state confirmation.

### Animation
Entrance motion is limited to a 180–240ms fade-and-rise sequence for workflow sheets. Provider toggles shift with a 160ms spring-like ease-out. Code blocks and copy confirmations respond instantly. All nonessential animation respects reduced-motion preferences.

### Typography System
Use **DM Mono** for project IDs, CLI commands, API fields, and metadata. Use **Manrope** for interface prose and controls, with a high-weight display treatment for page titles. Headlines are compact and direct; metadata remains restrained and letter-spaced.

### Brand Essence
**Nexuss-auth is a central authentication control plane for products and AI agents that need portable identity without operational friction.**

Personality: precise, calm, agent-native.

### Brand Voice
Headlines are directive and exact; CTAs use operational language rather than marketing filler. Examples: “Create a project, not another auth stack.” and “Enable the providers your users already trust.”

### Wordmark & Logo
Use the provided abstract three-orbit white mark set inside a black square beside the exact wordmark **Nexuss-auth** in a weighted geometric sans. The mark represents a protected identity moving through a trusted network.

### Signature Brand Color
**Absolute Signal Black — #050505.**

## Style Decisions

- The project canvas uses an offset operations-rail composition: project context is anchored left, while workflow sheets occupy an asymmetric workbench rather than a centered marketing column.
- Decorative line art must read as functional infrastructure notation — credential paths, routing, agents, or project topology — never as atmospheric sci-fi scenery.
- The brand lockup is always the three-orbit mark plus the exact wordmark **Nexuss-auth**; “Nexuss Auth” is not an acceptable alternate display form.


# Independent portfolio direction: Morrow Field

The portfolio is a separate brand called **Morrow Field**, presenting a fictional landscape and interaction designer who makes calm digital places for cultural organizations and thoughtful products. The visual language is warm editorial minimalism rather than authentication infrastructure: parchment, ink, mineral green, and small field-recording details.

## Design Movement
Contemporary editorial web design with Swiss grid discipline softened by field-journal imperfection.

## Core Principles
The page should feel authored, tactile, and easy to scan. Content uses asymmetric compositions instead of a centered marketing stack. Every interaction should reveal useful context, not decoration. The interface should feel personal without exposing implementation jargon.

## Color Philosophy
A warm bone background creates a quiet paper surface; ink-black type gives authority; mineral green signals active states and links to the landscape theme; muted clay marks selected work. The palette is intentionally low-saturation so project imagery and the signed-in identity state remain primary.

## Layout Paradigm
A narrow editorial rail anchors the left edge while the main work opens in a broad, offset column. Project cards read like annotated plates, with metadata placed beside rather than above the work. The signed-in state is a compact utility drawer rather than a dominant dashboard.

## Signature Elements
Use thin cartographic contour lines, small field-note labels, and an irregular vertical folio marker across the site. Motion should feel like paper and ink settling into place.

## Interaction Philosophy
The authentication controls should be clear and familiar, but visually belong to Morrow Field. After sign-in, the page should replace the prompt with the verified user returned by `getUser()`. Logout must restore the signed-out state and never fake identity locally.

## Animation
Use 180–260ms opacity and transform transitions. Stagger project plates by 50ms. Avoid layout animation. Respect reduced motion and keep sign-in actions immediate.

## Typography System
Use a serif display face for portfolio titles and a neutral sans-serif for interface text. The hierarchy is large, compact editorial headlines, small uppercase folio labels, and readable body copy with generous line height.

## Brand Essence
A quiet portfolio for people who want digital work with a sense of place; considered, observant, grounded.

## Brand Voice
Headlines are specific and sensory. CTAs are direct and calm. Avoid generic product language.

Example lines: “Interfaces with somewhere to go.” “Open the field notes.”

## Wordmark & Logo
The Morrow Field mark is a small offset square crossed by one contour line, paired with a custom-spaced wordmark. It should never resemble the Nexuss-auth orbit mark.

## Signature Brand Color
Mineral green: `#496B5A`.
