---
layout: single
entries_layout: grid
title: "Lectures"
collection: lectures
permalink: /lectures/
classes:
    - wide
---

The accompanying material for each lecture is posted here.

<div class="grid-collection-container">
    <div class="entries-{{ page.entries_layout }}">
    {% include documents-collection.html collection=page.collection sort_by=page.sort_by sort_order=page.sort_order type=page.entries_layout %}
    </div>
</div>

## Additional Resources

- The [supplemental material]({{ site.baseurl }}{% link _pages/supplements.md %}) page 
  contains prerequisite topics you should be familiar with.

- Detailed [notes]({{ site.baseurl }}{% link _pages/lecture_notes.md %}) are
  available for each lecture.
